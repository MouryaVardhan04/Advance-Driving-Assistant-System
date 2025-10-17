import numpy as np
import cv2
import pandas as pd
import pygame
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import Thresholding
from PerspectiveTransformation import PerspectiveTransformation
from LaneLines import LaneLines
from keras.models import load_model

class FindLaneLines:
    def __init__(self):
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

class RoadSignDetector:
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
            print(f"ERROR: Could not load model. Details: {e}")
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
        classIndex = np.argmax(predictions)
        probability = np.max(predictions) * 100
        predicted_name = self._get_class_name(classIndex)
        return predicted_name, probability

def overlay_sign_info(frame, name, confidence, pos=(None, 30)):
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

def run_sign_lane(input_path):
    lane_detector = FindLaneLines()
    sign_detector = RoadSignDetector()
    try:
        clip = VideoFileClip(input_path)
    except Exception as e:
        print(f"[SignLane] ERROR: Could not load video clip: {e}")
        return

    VIDEO_WIDTH, VIDEO_HEIGHT = clip.size

    pygame.init()
    screen = pygame.display.set_mode((VIDEO_WIDTH, VIDEO_HEIGHT))
    pygame.display.set_caption("Lane Detection & Road Sign Overlay")

    frame_count = 0
    print("\n[SignLane] --- Starting Integrated Video Processing ---")

    for frame in clip.iter_frames():
        frame_count += 1
        lane_out_frame = lane_detector.forward(frame)
        predicted_name, confidence = sign_detector.detect_frame(frame)

        # Overlay road sign info on top-right of the full lane output
        frame_overlay = lane_out_frame.copy()
        frame_overlay = overlay_sign_info(frame_overlay, predicted_name, confidence, pos=(VIDEO_WIDTH-300, 30))

        if frame_count % 30 == 0:
            print(f"[SignLane] Frame {frame_count:04d} | Sign: {predicted_name} | Conf: {confidence:.2f}%")
        frame_rgb = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(frame_pygame, (0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print("[SignLane] Closed by user.")
                return

    pygame.quit()
    print("[SignLane] Video processing finished.")

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "project_video.mp4"
    run_sign_lane(input_path)