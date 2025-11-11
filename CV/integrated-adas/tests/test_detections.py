import unittest
from src.detection.lane_detection import LaneDetector
from src.detection.sign_detection import SignDetector
from src.detection.drowsiness_detection import DrowsinessDetector
from src.detection.emotion_detection import EmotionDetector

class TestDetections(unittest.TestCase):

    def setUp(self):
        self.lane_detector = LaneDetector()
        self.sign_detector = SignDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.emotion_detector = EmotionDetector()

    def test_lane_detection(self):
        # Assuming we have a test image for lane detection
        test_image = 'path/to/test/lane_image.jpg'
        lanes = self.lane_detector.detect_lanes(test_image)
        self.assertIsNotNone(lanes)
        self.assertGreater(len(lanes), 0)

    def test_sign_detection(self):
        # Assuming we have a test image for sign detection
        test_image = 'path/to/test/sign_image.jpg'
        signs = self.sign_detector.detect_signs(test_image)
        self.assertIsNotNone(signs)
        self.assertGreater(len(signs), 0)

    def test_drowsiness_detection(self):
        # Assuming we have a test image for drowsiness detection
        test_image = 'path/to/test/drowsiness_image.jpg'
        drowsiness_status = self.drowsiness_detector.check_drowsiness(test_image)
        self.assertIn(drowsiness_status, ['alert', 'drowsy'])

    def test_emotion_detection(self):
        # Assuming we have a test image for emotion detection
        test_image = 'path/to/test/emotion_image.jpg'
        emotion = self.emotion_detector.detect_emotion(test_image)
        self.assertIsNotNone(emotion)

if __name__ == '__main__':
    unittest.main()