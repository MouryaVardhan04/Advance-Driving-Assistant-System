class EmotionDetector:
    def __init__(self, model_path='src/models/emotion_model.h5'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        from keras.models import load_model
        try:
            model = load_model(self.model_path)
            return model
        except Exception as e:
            print(f"ERROR: Could not load emotion detection model. Details: {e}")
            return None

    def preprocess_image(self, img):
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))  # Assuming the model expects 48x48 input
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 48, 48, 1)  # Reshape for the model
        return img

    def detect_emotion(self, img):
        processed_img = self.preprocess_image(img)
        predictions = self.model.predict(processed_img)
        emotion_index = predictions.argmax()
        return emotion_index, predictions[0][emotion_index]  # Return the index and confidence