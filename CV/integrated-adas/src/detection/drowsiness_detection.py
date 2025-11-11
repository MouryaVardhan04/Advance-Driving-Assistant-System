class DrowsinessDetector:
    def __init__(self, model_path='src/models/drowsiness_model.h5'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        from keras.models import load_model
        try:
            return load_model(self.model_path)
        except Exception as e:
            print(f"ERROR: Could not load drowsiness detection model. Details: {e}")
            return None

    def preprocess_frame(self, frame):
        # Preprocess the frame for drowsiness detection
        # This is a placeholder for actual preprocessing logic
        return frame

    def check_drowsiness(self, frame):
        preprocessed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(preprocessed_frame)
        # Assuming the model outputs a probability of drowsiness
        return prediction[0][0] > 0.5  # Example threshold for drowsiness detection

    def annotate_frame(self, frame, is_drowsy):
        # Annotate the frame with drowsiness detection results
        label = "Drowsy" if is_drowsy else "Alert"
        import cv2
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not is_drowsy else (0, 0, 255), 2)
        return frame