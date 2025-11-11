import time
import cv2
import numpy as np
import pygame
import supervision as sv
from ultralytics import YOLO

# ---- Config ----
MODEL_PATH = "best.pt"    # path to your model
VIDEO_PATH = "demo.mp4"   # path to video
WINDOW_TITLE = "PyResearch - Pothole Detection (Pygame)"
WINDOW_SIZE = (1280, 720) # optional initial size (we will scale frames to fit)
RESIZE_TO = (960, 540)    # frame resize for smoother display (set None to keep original)
FONT_SIZE = 18

# ---- Globals ----
detection_count = 0

# ---- Initialize model & annotator ----
print("Loading model...")
model = YOLO(MODEL_PATH)

box_annotator = sv.BoxAnnotator(
    thickness=2,
    color=sv.Color.from_hex("#0055FF")
)

print("Model loaded.")

# ---- Helper: annotate frame and return BGR image ----
def annotate_frame_bgr(frame_bgr):
    """
    Run model inference on BGR frame, return annotated BGR frame and detection_count.
    """
    global detection_count

    # Ultralytics expects RGB or BGR depending on model; passing BGR (cv2) is okay for many versions.
    results = model(frame_bgr)[0]

    # Convert ultralytics results -> supervision Detections
    detections = sv.Detections.from_ultralytics(results)

    # detection_count for dashboard
    detection_count = len(detections)

    # Build labels (results.names + conf)
    # detections.boxes data layout: [x1, y1, x2, y2, conf, class]
    # However supervision.Detections exposes convenience attributes: xyxy, confidence, class_id
    try:
        labels = [
            f"{results.names[int(cls)]}: {float(conf):.2f}"
            for conf, cls in zip(detections.confidence, detections.class_id)
        ]
    except Exception:
        # fallback if zipped attributes mismatch
        labels = []
        for det in detections:
            # try to index [4]=conf, [5]=class if it's a raw row
            try:
                conf = float(det[4])
                cls = int(det[5])
                labels.append(f"{results.names[cls]}: {conf:.2f}")
            except Exception:
                labels.append("obj")

    # annotate (BoxAnnotator expects positional args in newer supervision)
    annotated = box_annotator.annotate(frame_bgr, detections, labels)

    return annotated, detection_count

# ---- Pygame setup ----
pygame.init()
pygame.display.set_caption(WINDOW_TITLE)
screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", FONT_SIZE)

# ---- OpenCV video capture ----
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open video file {VIDEO_PATH}")

# Optionally compute frame scaling to fit window while preserving aspect ratio
def scale_frame_to_window(frame, window_size):
    fh, fw = frame.shape[:2]
    ww, wh = window_size
    scale = min(ww / fw, wh / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized

# ---- Main loop ----
running = True
last_time = time.time()
fps = 0.0
frame_counter = 0
fps_calc_interval = 0.5
fps_timer = time.time()

print("Starting main loop. Press window close or ESC to quit.")
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.VIDEORESIZE:
            # update screen size when user resizes window
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

    success, frame = cap.read()
    if not success:
        # loop video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Optionally resize input frame to speed up inference
    if RESIZE_TO:
        frame = cv2.resize(frame, RESIZE_TO)

    # Annotate (returns BGR)
    try:
        annotated_bgr, det_count = annotate_frame_bgr(frame)
    except Exception as e:
        # If annotation/inference fails, print and continue with original frame
        print("Inference/annotate error:", e)
        annotated_bgr = frame.copy()
        det_count = 0

    # Convert BGR -> RGB for pygame, then to surface
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Fit annotated frame into current window, keeping aspect ratio
    window_size = pygame.display.get_surface().get_size()
    display_img = scale_frame_to_window(annotated_rgb, window_size)

    # Convert to pygame surface
    surf = pygame.image.frombuffer(display_img.tobytes(), display_img.shape[1::-1], "RGB")

    # Blit to screen (centered)
    screen.fill((20, 20, 20))
    rect = surf.get_rect(center=(window_size[0] // 2, window_size[1] // 2))
    screen.blit(surf, rect.topleft)

    # Compute FPS (smoothed)
    frame_counter += 1
    now = time.time()
    if now - fps_timer >= fps_calc_interval:
        fps = frame_counter / (now - fps_timer)
        fps_timer = now
        frame_counter = 0

    # Overlay FPS and detection count using pygame font
    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
    det_text = font.render(f"Detections: {det_count}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))
    screen.blit(det_text, (10, 10 + FONT_SIZE + 6))

    # Optionally display model name / info
    model_text = font.render(f"Model: {MODEL_PATH}", True, (200, 200, 200))
    screen.blit(model_text, (10, window_size[1] - FONT_SIZE - 10))

    pygame.display.flip()

    # limit to ~30-60hz (adjust as required)
    clock.tick(60)

# Cleanup
cap.release()
pygame.quit()
print("Exited cleanly.")
