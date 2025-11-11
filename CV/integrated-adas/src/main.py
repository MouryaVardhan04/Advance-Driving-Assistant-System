import cv2
from detection.lane_detection import LaneDetector
from detection.sign_detection import SignDetector
from detection.drowsiness_detection import DrowsinessDetector
from detection.emotion_detection import EmotionDetector
from utils.camera_calibration import CameraCalibration
from utils.visualization import overlay_results

def main():
    # Initialize detectors
    lane_detector = LaneDetector()
    sign_detector = SignDetector()
    drowsiness_detector = DrowsinessDetector()
    emotion_detector = EmotionDetector()

    # Camera calibration
    camera_calibration = CameraCalibration('camera_cal', 9, 6)

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame
        frame = camera_calibration.undistort(frame)

        # Lane detection
        lane_image = lane_detector.detect_lanes(frame)

        # Sign detection
        sign_name, sign_confidence = sign_detector.detect_signs(frame)

        # Drowsiness detection
        drowsiness_status = drowsiness_detector.check_drowsiness(frame)

        # Emotion detection
        emotion, emotion_confidence = emotion_detector.detect_emotion(frame)

        # Overlay results
        result_image = overlay_results(lane_image, sign_name, sign_confidence, drowsiness_status, emotion, emotion_confidence)

        # Display the result
        cv2.imshow('Integrated ADAS', result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()