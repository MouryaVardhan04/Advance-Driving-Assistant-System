class SignDetector:
    def __init__(self, model_path='src/models/sign_model.h5', label_file='src/config/labels.csv'):
        self.model_path = model_path
        self.label_file = label_file
        self.model = self._load_model()
        self.class_names = self._load_labels()

    def _load_model(self):
        from keras.models import load_model
        try:
            return load_model(self.model_path)
        except Exception as e:
            print(f"ERROR: Could not load model. Details: {e}")
            return None

    def _load_labels(self):
        import pandas as pd
        try:
            data = pd.read_csv(self.label_file)
            data.columns = data.columns.str.strip()
            return dict(zip(data['ClassId'], data['Name']))
        except Exception as e:
            print(f"ERROR: Could not load labels. Details: {e}")
            return {}

    def detect_signs(self, image):
        # Preprocess the image for the model
        processed_image = self._preprocess_image(image)
        predictions = self.model.predict(processed_image)
        class_index = predictions.argmax()
        confidence = predictions.max()
        sign_name = self.class_names.get(class_index, "UNKNOWN")
        return sign_name, confidence

    def _preprocess_image(self, image):
        import cv2
        image = cv2.resize(image, (32, 32))  # Resize to model input size
        image = image.astype('float32') / 255.0  # Normalize the image
        return image.reshape(1, 32, 32, 3)  # Reshape for the model input