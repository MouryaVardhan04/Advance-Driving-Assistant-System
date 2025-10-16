import numpy as np
import cv2
import pandas as pd
from keras.models import load_model


class RoadSignDetector:
    def __init__(self, model_path='Models/sign_model.h5', label_file='labels.csv', video_path='roadsign.webm'):
        self.model_path = model_path
        self.label_file = label_file
        self.video_path = video_path

        # Image dimensions (must match training)
        self.IMG_WIDTH, self.IMG_HEIGHT = 32, 32

        # Load model + labels
        self.model = self._load_model()
        self.classNames = self._load_labels()

    # ---------------- Utility Functions ----------------
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

    # ---------------- Loaders ----------------
    def _load_model(self):
        print(f"Loading model from: {self.model_path}")
        try:
            return load_model(self.model_path)
        except Exception as e:
            print(f"ERROR: Could not load model. Details: {e}")
            raise

    def _load_labels(self):
        print(f"Loading class labels from: {self.label_file}")
        try:
            data = pd.read_csv(self.label_file)
            data.columns = data.columns.str.strip()
            classNames = dict(zip(data['ClassId'], data['Name']))
            print(f"Detected {len(classNames)} classes.")
            return classNames
        except Exception as e:
            print(f"ERROR: Could not load labels. Details: {e}")
            raise

    # ---------------- Main Detection ----------------
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video file: {self.video_path}")
            return

        frame_count = 0
        print("\n--- Starting Video Prediction ---")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            img = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
            processed_img = self._preprocessing(img)

            predictions = self.model.predict(processed_img, verbose=0)
            classIndex = np.argmax(predictions)
            probability = np.max(predictions) * 100
            predicted_name = self._get_class_name(classIndex)

            # Box & Text
            BOX_X_START, BOX_Y_START = 10, 10
            BOX_WIDTH, BOX_HEIGHT = 220, 90
            BOX_COLOR = (0, 255, 0) if probability > 80 else (0, 165, 255)

            cv2.rectangle(frame, (BOX_X_START, BOX_Y_START),
                          (BOX_X_START + BOX_WIDTH, BOX_Y_START + BOX_HEIGHT),
                          BOX_COLOR, 2)

            cv2.putText(frame, f"Sign: {predicted_name}",
                        (BOX_X_START + 10, BOX_Y_START + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Conf: {probability:.2f}%",
                        (BOX_X_START + 10, BOX_Y_START + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if frame_count % 10 == 0:
                print(f"Frame {frame_count:04d} | Prediction: {predicted_name} | Confidence: {probability:.2f}%")

            cv2.imshow("Road Sign Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n--- Video Prediction Finished ---")


# ---------------- Public API ----------------
def run_detector():
    detector = RoadSignDetector()
    detector.run()
