import numpy as np
import cv2
import pandas as pd
import pygame
from keras.models import load_model
import time # <--- Added for integration

# Pothole/YOLO Imports
import supervision as sv
from ultralytics import YOLO

# Assuming these classes exist in separate files or are defined above this script
from CameraCalibration import CameraCalibration
from Thresholding import Thresholding
from PerspectiveTransformation import PerspectiveTransformation
from LaneLines import LaneLines # MUST be the version with integrated pothole logic
from Human_Detection import HumanDetector  # Add human detection import

# ---- Pothole Detection Config (Copied from pothole.py for reference) ----
# Note: The actual model loading and config is handled within the LaneLines class's __init__
MODEL_PATH = "best.pt" 
CONFIDENCE_THRESHOLD = 0.10  # Lowered from 0.70 to match working debug_pothole.py (which uses 0.1) 
# -------------------------------------------------------------------------


class FindLaneLines:
    """
    Combines Camera Calibration, Thresholding, Perspective Transformation,
    and LaneLines (which includes Pothole Detection logic).
    """
    def __init__(self):
        # Initializations for Lane Detection
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        
        # LaneLines class now handles Pothole Model initialization
        # Pass model path and confidence threshold from config
        self.lanelines = LaneLines(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
        
        # Initialize human detector
        self.human_detector = HumanDetector()

    def forward(self, img):
        """Processes the frame for lane detection and returns the base annotated image."""
        
        # 1. Undistort and setup copies
        out_img = np.copy(img)
        undistorted_img = self.calibration.undistort(img)
        
        # 2. Lane detection pipeline
        warped_img = self.transform.forward(undistorted_img)
        binary_img = self.thresholding.forward(warped_img)
        lane_drawn_img = self.lanelines.forward(binary_img) # Runs detection/fit, returns green lane overlay
        
        # 3. Only add lane overlay if lanes were successfully detected and validated
        if getattr(self.lanelines, 'lanes_valid', False):
            # Unwarp lane overlay
            unwarped_lane_overlay = self.transform.backward(lane_drawn_img)
            # Combine original image and lane overlay
            final_img = cv2.addWeighted(undistorted_img, 1, unwarped_lane_overlay, 0.6, 0)
        else:
            # No lane detection - just use the undistorted image without overlay
            final_img = undistorted_img.copy()
        
        # 5. Store original color frame for pothole detection
        # The LaneLines.plot method will use this frame for pothole detection
        self.lanelines.set_original_frame(undistorted_img)
        
        # 6. Plot overlays (Lane info widget + Pothole warning)
        # The LaneLines.plot method runs pothole detection on the original color frame
        # and draws bounding boxes and warnings if potholes are detected.
        final_img = self.lanelines.plot(final_img)
        
        # 7. Run human detection on the undistorted image
        human_boxes, human_confidences = self.human_detector.detect(undistorted_img)
        
        # 8. Draw human detection boxes
        if human_boxes is not None and len(human_boxes) > 0:
            final_img = self.human_detector.draw_detections(final_img, human_boxes, human_confidences)
        
        return final_img


class RoadSignDetector:
    # (RoadSignDetector class logic is unchanged)
    def __init__(self, model_path='Models/sign_model.h5', label_file='labels.csv'):
        self.model_path = model_path
        self.label_file = label_file
        self.IMG_WIDTH, self.IMG_HEIGHT = 32, 32
        self.model = self._load_model()
        self.classNames = self._load_labels()

    def _grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _equalize(self, img):
        return cv2.equalizeHist(img)

    def _preprocessing(self, img):
        img = self._grayscale(img)
        img = self._equalize(img)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, self.IMG_WIDTH, self.IMG_HEIGHT, 1)
        return img

    def _get_class_name(self, classNo):
        return self.classNames.get(classNo, "UNKNOWN CLASS")

    def _load_model(self):
        try:
            return load_model(self.model_path)
        except Exception as e:
            print(f"ERROR: Could not load sign model. Details: {e}")
            class DummyModel:
                def predict(self, x, verbose=0):
                    return np.array([[0.0, 0.0, 1.0]])
            return DummyModel()

    def _load_labels(self):
        try:
            data = pd.read_csv(self.label_file)
            data.columns = data.columns.str.strip()
            classNames = dict(zip(data['ClassId'], data['Name']))
            return classNames
        except Exception as e:
            print(f"ERROR: Could not load labels. Details: {e}")
            return {0: "Speed Limit", 1: "Stop", 2: "Yield"}

    def detect_frame(self, frame):
        img = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
        processed_img = self._preprocessing(img)
        predictions = self.model.predict(processed_img, verbose=0)
        classIndex = int(np.argmax(predictions))
        probability = float(np.max(predictions) * 100)
        predicted_name = self._get_class_name(classIndex)
        # Return class index as well so caller can map to an icon file without re-parsing labels
        return classIndex, predicted_name, probability

def overlay_sign_info(frame, name, confidence, pos=(None, 30)):
    # (overlay_sign_info logic is unchanged)
    h, w = frame.shape[:2]
    x = w - 320 if pos[0] is None else pos[0]
    y = pos[1]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_COLOR = (255, 255, 255)
    CONF_COLOR = (0, 255, 0) if confidence > 80 else (0, 165, 255)
    cv2.putText(frame, "ROAD SIGN DETECTOR", (x, y), FONT, 0.7, TEXT_COLOR, 2)
    cv2.putText(frame, f"Sign: {name}", (x, y+40), FONT, 0.6, CONF_COLOR, 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}%", (x, y+80), FONT, 0.7, CONF_COLOR, 2)
    return frame


def overlay_sign_visual(frame, class_index, name, confidence, asserts_dir='asserts', icon_size=(120, 120), margin=20):
    """Overlay a sign icon inside a widget at the top-right when confidence >= 75%.
    If confidence < 75% nothing is drawn (no text), preserving the original frame.
    The widget contains a semi-transparent background, a 1px boundary, the icon centered
    vertically in the widget, and a small label centered at the bottom of the widget.
    """
    try:
        if confidence < 75.0:
            # Do not display anything if confidence is below threshold
            return frame

        h, w = frame.shape[:2]

        # Widget configuration
        widget_w, widget_h = 220, 160
        widget_x = w - widget_w - margin
        widget_y = margin

        # Draw semi-transparent background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (widget_x, widget_y), (widget_x + widget_w, widget_y + widget_h), (30, 30, 30), -1)
        alpha = 0.55
        frame[widget_y:widget_y+widget_h, widget_x:widget_x+widget_w] = cv2.addWeighted(
            overlay[widget_y:widget_y+widget_h, widget_x:widget_x+widget_w],
            alpha,
            frame[widget_y:widget_y+widget_h, widget_x:widget_x+widget_w],
            1 - alpha,
            0
        )

        # Draw boundary box (1px)
        cv2.rectangle(frame, (widget_x, widget_y), (widget_x + widget_w, widget_y + widget_h), (200, 200, 200), 1)

        # Load icon image
        icon_path = f"{asserts_dir}/{class_index}.png"
        icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            # Icon missing - don't display text either; just return frame
            return frame

        # Compute icon size to fit inside widget (leave space for label)
        max_icon_w = widget_w - 40
        max_icon_h = widget_h - 60
        ih, iw = icon.shape[:2]
        scale = min(max_icon_w / iw, max_icon_h / ih, 1.0)
        icon_w, icon_h = int(iw * scale), int(ih * scale)
        icon_resized = cv2.resize(icon, (icon_w, icon_h), interpolation=cv2.INTER_AREA)

        # Center icon horizontally and vertically (reserve bottom area for label)
        ix = widget_x + (widget_w - icon_w) // 2
        iy = widget_y + (widget_h - icon_h - 30) // 2

        # Ensure ROI within frame boundaries
        if iy < 0 or ix < 0 or iy + icon_h > h or ix + icon_w > w:
            return frame

        roi = frame[iy:iy+icon_h, ix:ix+icon_w]

        # Blend icon if it has alpha channel
        if icon_resized.shape[2] == 4:
            alpha_mask = icon_resized[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = (alpha_mask * icon_resized[:, :, c] + (1 - alpha_mask) * roi[:, :, c]).astype('uint8')
            frame[iy:iy+icon_h, ix:ix+icon_w] = roi
        else:
            frame[iy:iy+icon_h, ix:ix+icon_w] = icon_resized

        # Draw label centered at the bottom of the widget
        label = name
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, FONT, font_scale, thickness)
        tx = widget_x + (widget_w - text_w) // 2
        ty = widget_y + widget_h - 12
        cv2.putText(frame, label, (tx, ty), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame
    except Exception:
        # On error, do nothing (no overlay)
        return frame

def run_sign_lane(input_path):
    # Initialize detectors (LaneLines handles Pothole initialization)
    lane_detector = FindLaneLines()
    sign_detector = RoadSignDetector()
    
    # Open video using OpenCV (much faster than moviepy)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[SignLane] ERROR: Could not open video file: {input_path}")
        return
    
    # Get video properties
    VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n[SignLane] --- Starting Integrated Video Processing ---")

    pygame.init()
    screen = pygame.display.set_mode((VIDEO_WIDTH, VIDEO_HEIGHT))
    pygame.display.set_caption("Lane Detection & Road Sign Overlay")

    frame_count = 0
    start_time = time.time()

    running = True
    while running:
        
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                break
        
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            # End of video or error
            break
        
        frame_count += 1
        
        # Convert BGR to RGB (OpenCV uses BGR, moviepy uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Run Lane/Pothole Pipeline
        lane_out_frame = lane_detector.forward(frame_rgb)
        
        # 2. Run Road Sign Detection (now returns class index too)
        class_idx, predicted_name, confidence = sign_detector.detect_frame(frame_rgb)

        # 3. Overlay road sign icon/text
        frame_overlay = lane_out_frame.copy()
        frame_overlay = overlay_sign_visual(frame_overlay, class_idx, predicted_name, confidence, asserts_dir='asserts')
        
        # 4. FPS Calculation and Overlay
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_fps = frame_count / elapsed_time
        else:
            current_fps = 0.0

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_overlay, f"FPS: {current_fps:.1f}", (10, 30), FONT, 0.7, (255, 255, 255), 2)

        if frame_count % 30 == 0:
            print(f"[SignLane] Frame {frame_count:04d} | Sign: {predicted_name} | Conf: {confidence:.2f}%")
        
        # 5. Display frame via Pygame
        # Convert BGR to RGB for pygame
        frame_display = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.surfarray.make_surface(frame_display.swapaxes(0, 1))
        screen.blit(frame_pygame, (0, 0))
        pygame.display.flip()

    cap.release()
    pygame.quit()
    print("[SignLane] Video processing finished.")

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "project_video.mp4"
    run_sign_lane(input_path)