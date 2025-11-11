import cv2
from Human_Detection import Detector
# from Human_Detection import Detector # Assuming 'Human_Detection.py' is in the same directory

cap = cv2.VideoCapture('demo.mp4')

# --- Optimization Parameters ---
# Process the detection function only every 'skip_frames' frame
skip_frames = 3 
frame_count = 0
display_frame = None # Variable to hold the last frame with detection

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video stream.")
        break

    # 1. Resize the frame once (crucial for speed)
    frame = cv2.resize(frame, (640, 480))
    
    if frame_count % skip_frames == 0:
        # 2. Heavy Processing: Only run the Detector periodically
        processed_frame = Detector(frame.copy()) # Use a copy to prevent in-place modification issues
        display_frame = processed_frame
    else:
        # 3. Light Processing: Display the result from the last detection frame
        if display_frame is None:
             # Handle the very first frame where display_frame is not yet set
             display_frame = frame 
        else:
            # We already have a frame with bounding boxes from a previous run, 
            # so we can use the original frame and draw the old boxes, 
            # OR just display the last fully processed frame. 
            # For simplicity and speed, we'll just display the last detected frame for now.
            pass

    # Ensure display_frame is set before attempting to show
    if display_frame is not None:
        cv2.imshow('Car Detection System - Optimized Speed', display_frame)

    frame_count += 1

    # cv2.resizeWindow('Car Detection System', 600, 600)
    k = cv2.waitKey(1) & 0xff
    if k == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()