from tensorflow.keras.models import load_model # Using the correct import path
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array # Using the correct import path
import cv2
import numpy as np
import os # Necessary for safer path construction

# --- Setup ---
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# CRITICAL FIX: Load the correct model file using a proper string path
classifier = load_model(os.path.join('Models', 'emotion_model.h5')) 

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    
    # Check if the frame was read successfully
    if frame is None:
        print("Error: Could not read frame from camera.")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improved face detection parameters to reduce false positives (like the switch)
    faces = face_classifier.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=4, 
        minSize=(40, 40)
    )

    # --- Logic to Detect ONLY the Largest Face ---
    if len(faces) > 0:
        # Sort faces by area (w * h) in descending order
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        # Select the largest face (x, y, w, h)
        (x, y, w, h) = faces[0]
        
        # Draw bounding box on the largest face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Prepare ROI for classification
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            
            # CRITICAL FIX: Add batch and channel dimensions for the model (1, 48, 48, 1)
            roi = np.expand_dims(roi, axis=0)  # Shape (1, 48, 48)
            roi = np.expand_dims(roi, axis=-1) # Final Shape (1, 48, 48, 1)

            # Predict emotion
            prediction = classifier.predict(roi, verbose=0)[0] 
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            
            # Display emotion label
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # This should technically not be reached if faces[0] was valid, but kept for safety
            cv2.putText(frame, 'No Valid Face', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # No faces detected at all
        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow('Emotion Detector', frame)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()