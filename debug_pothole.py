import sys
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "best.pt"
VIDEO_PATH = "demo.mp4"
OUT_IMAGE = "debug_output.jpg"

def draw_boxes_on_frame(frame, boxes_xyxy, confs=None, cls_ids=None, names=None):
    # boxes_xyxy: Nx4 array
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        conf = float(confs[i]) if confs is not None and len(confs) > i else None
        cls = int(cls_ids[i]) if cls_ids is not None and len(cls_ids) > i else None
        label = f"{names.get(cls, cls) if names else cls}:{conf:.2f}" if (names is not None and conf is not None) else str(cls)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # open video and grab a frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video. You can also pass an image path as argument.")
        if len(sys.argv) > 1:
            img_path = sys.argv[1]
            frame = cv2.imread(img_path)
            if frame is None:
                print("Failed to read image", img_path)
                return
        else:
            return
    else:
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to read frame from video")
            return

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Convert to RGB for model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference with low conf for debugging
    print("Running inference (conf=0.1)...")
    try:
        results = model.predict(source=frame_rgb, conf=0.1, iou=0.45, verbose=False)[0]
    except Exception as e:
        print("model.predict failed, trying model(frame_rgb)", e)
        results = model(frame_rgb)[0]

    # Print results summary
    print("Results type:", type(results))
    names = getattr(results, 'names', None)
    print("Names mapping:", names)

    boxes = None
    confs = None
    cls = None

    if hasattr(results, 'boxes'):
        boxes_obj = results.boxes
        # Try common attributes
        try:
            xyxy = boxes_obj.xyxy.cpu().numpy()
            confs = boxes_obj.conf.cpu().numpy()
            cls = boxes_obj.cls.cpu().numpy().astype(int)
            boxes = xyxy
            print(f"Found {len(boxes)} boxes via boxes.xyxy")
        except Exception as e:
            print("Could not read boxes via typical attributes:", e)
            # try alternative
            try:
                xyxy = boxes_obj.xyxy
                boxes = np.array(xyxy)
                print(f"Found {len(boxes)} boxes via boxes.xyxy (no .cpu)")
            except Exception as e2:
                print("No box coords available on results.boxes:", e2)

    # Some ultralytics versions store boxes in results.xyxyn or results.masks etc.
    if boxes is None:
        # Try to inspect results manually
        try:
            print("results.raw:")
            print(dir(results))
            if hasattr(results, 'xyxy'):
                boxes = np.array(results.xyxy)
                print(f"Found {len(boxes)} in results.xyxy")
        except Exception as e:
            print("No fallback boxes found:", e)

    if boxes is None or len(boxes) == 0:
        print("No boxes detected by model at chosen conf. Try lowering conf or testing on a known pothole image.")
        # Save an image for reference
        cv2.imwrite(OUT_IMAGE, frame)
        print("Saved frame to", OUT_IMAGE)
        return

    # Draw boxes manually and save
    out = draw_boxes_on_frame(frame.copy(), boxes, confs, cls, names if isinstance(names, dict) else None)
    cv2.imwrite(OUT_IMAGE, out)
    print("Wrote debug output to", OUT_IMAGE)
    # Also print box numeric values
    for i, b in enumerate(boxes):
        c = confs[i] if confs is not None and len(confs) > i else None
        cl = int(cls[i]) if cls is not None and len(cls) > i else None
        print(f"Box[{i}]: {b}, conf={c}, cls={cl}, name={names.get(cl) if isinstance(names, dict) else cl}")

if __name__ == '__main__':
    main()
