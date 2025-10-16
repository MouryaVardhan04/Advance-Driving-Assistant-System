import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os # For safer path construction

# ==============================================================================
# --- DROWSINESS DETECTION COMPONENTS (Constants, Functions, Class) ---
# ==============================================================================

# --- Configuration Constants ---
EAR_THRESHOLD = 0.25      
MAR_THRESHOLD = 0.6       
TERMINAL_OUTPUT_INTERVAL = 2  

# --- Time-Based Alert Constants ---
EAR_DURATION_ALERT_SEC = 2.0 
MAR_DURATION_ALERT_SEC = 4.0 

# --- Recovery Logic Constants ---
RECOVERY_TIME_SEC = 20.0 

# --- Custom Alert Logic Constants ---
EAR_ALERT_LIMIT_L1 = 3  
MAR_ALERT_LIMIT_L1 = 2  
EAR_ALERT_LIMIT_L2 = 5  
MAR_ALERT_LIMIT_L2 = 3  

# Landmark Indices
R_EYE_IDXS = [33, 160, 158, 133, 153, 144] 
L_EYE_IDXS = [362, 385, 387, 263, 373, 380] 
MOUTH_IDXS = [61, 291, 0, 17, 14, 37, 267, 40, 270, 310, 317, 82, 312]

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

# --- Helper Functions ---
def _eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) 
    B = dist.euclidean(eye[2], eye[4]) 
    C = dist.euclidean(eye[0], eye[3]) 
    ear = (A + B) / (2.0 * C)
    return ear

def _mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[11]) 
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[3], mouth[9])
    D = dist.euclidean(mouth[0], mouth[6]) 
    mar = (A + B + C) / (3.0 * D)
    return mar

# --- Drowsiness Detector Class ---
class DrowsinessDetector:
    """Detects signs of drowsiness (eye closure and yawning) from a video frame."""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Time and Event Tracking
        self.EAR_CLOSED_START_TIME = None  
        self.MAR_YAWN_START_TIME = None    
        self.EAR_ALERT_CONFIRMED = False   
        self.MAR_ALERT_CONFIRMED = False

        # Cumulative Counters and Status
        self.EAR_ALERT_COUNT = 0 
        self.MAR_ALERT_COUNT = 0 
        self.SLEEPINESS_LEVEL = 0
        self.last_terminal_output = 0
        
        # Recovery Timer Tracking
        self.recovery_countdown_active = False
        self.alert_start_time = time.time()
        self.last_drowsy_level = 0 

    def _update_sleepiness_level(self):
        """Calculates the sleepiness level based on cumulative EAR/MAR counts."""
        ear_count = self.EAR_ALERT_COUNT
        mar_count = self.MAR_ALERT_COUNT

        if ear_count > EAR_ALERT_LIMIT_L2 and mar_count > MAR_ALERT_LIMIT_L2:
            return 3
        elif ear_count >= EAR_ALERT_LIMIT_L2 and mar_count >= MAR_ALERT_LIMIT_L2:
            return 2
        elif ear_count >= EAR_ALERT_LIMIT_L1 and mar_count >= MAR_ALERT_LIMIT_L1:
            return 1
        else:
            return 0 

    def _apply_recovery_logic(self, current_level, avg_ear, mar):
        """
        Decrements EAR and MAR counts by 1 after 20 seconds of continuous alertness.
        """
        current_time = time.time()
        is_frame_alert = (avg_ear >= EAR_THRESHOLD) and (mar <= MAR_THRESHOLD)
        
        if current_level > 0:
            self.recovery_countdown_active = False
            self.alert_start_time = current_time 
            self.last_drowsy_level = current_level
            
        elif is_frame_alert and (self.EAR_ALERT_COUNT > 0 or self.MAR_ALERT_COUNT > 0):
            if not self.recovery_countdown_active:
                self.recovery_countdown_active = True
                self.alert_start_time = current_time 
            
            if self.recovery_countdown_active and (current_time - self.alert_start_time >= RECOVERY_TIME_SEC):
                self.EAR_ALERT_COUNT = max(0, self.EAR_ALERT_COUNT - 1)
                self.MAR_ALERT_COUNT = max(0, self.MAR_ALERT_COUNT - 1)
                
                print(f"{Colors.GREEN}[{datetime.now().strftime('%H:%M:%S')}] 🧠 RECOVERY: Alertness held for {RECOVERY_TIME_SEC}s. Counts decremented by 1. New Counts (E:{self.EAR_ALERT_COUNT}, M:{self.MAR_ALERT_COUNT}){Colors.END}")

                self.alert_start_time = current_time 
        else:
            self.recovery_countdown_active = False 
            self.alert_start_time = current_time 

    def _print_terminal_alert(self, level, ear, mar):
        """Prints simplified, colored terminal output for drowsiness alerts."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level > 0 or time.time() - self.last_terminal_output >= TERMINAL_OUTPUT_INTERVAL:
            common_stats = f" | Counts (EAR: {self.EAR_ALERT_COUNT}, MAR: {self.MAR_ALERT_COUNT}) | Metrics (EAR: {ear:.3f}, MAR: {mar:.3f})"
            
            if level == 3:
                print(f"{Colors.RED}{Colors.BOLD}[{timestamp}] 🚨 DEEP SLEEP DETECTED! (Level 3) 🚨{Colors.END}")
            elif level == 2:
                print(f"{Colors.YELLOW}{Colors.BOLD}[{timestamp}] ⚠️  MEDIUM SLEEP DETECTED! (Level 2) ⚠️{Colors.END}")
            elif level == 1:
                print(f"{Colors.CYAN}[{timestamp}] 😴 NORMAL SLEEP DETECTED! (Level 1){Colors.END}")
            else:
                if self.recovery_countdown_active:
                    time_left = max(0, RECOVERY_TIME_SEC - (time.time() - self.alert_start_time))
                    print(f"{Colors.GREEN}[{timestamp}] ✅ Alert - Level 0 (Recovery Time Left: {time_left:.0f}s){common_stats}{Colors.END}")
                else:
                    print(f"{Colors.GREEN}[{timestamp}] ✅ Fully Alert{common_stats}{Colors.END}")

            self.last_terminal_output = time.time()

    def _draw_display_status(self, image, img_w, ear_metric, mar_metric):
        """Draws the status box on the image."""
        
        status_color = (0, 255, 0)
        status_text = f"LEVEL 0: ALERT"

        if self.SLEEPINESS_LEVEL == 3:
            status_text = "LEVEL 3: DEEP SLEEP"
            status_color = (0, 0, 255) 
        elif self.SLEEPINESS_LEVEL == 2:
            status_text = "LEVEL 2: MEDIUM SLEEP"
            status_color = (0, 165, 255) 
        elif self.SLEEPINESS_LEVEL == 1:
            status_text = "LEVEL 1: NORMAL SLEEP"
            status_color = (0, 255, 255) 
        
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(image, f"EAR Count: {self.EAR_ALERT_COUNT}", (img_w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"MAR Count: {self.MAR_ALERT_COUNT}", (img_w - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return self._run_drowsiness_logic(image)

    def _run_drowsiness_logic(self, image):
        """Processes the frame for Drowsiness, returns metrics and annotated image."""
        
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        image.flags.writeable = True
        
        img_h, img_w, _ = image.shape
        ear_alert_instant, mar_alert_instant = False, False 
        avg_ear, mar = 0, 0
        current_time = time.time()
        
        face_detected = False
        face_coords = None

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Get bounding box for emotion detection later
            x_min = min(int(lm.x * img_w) for lm in landmarks)
            y_min = min(int(lm.y * img_h) for lm in landmarks)
            x_max = max(int(lm.x * img_w) for lm in landmarks)
            y_max = max(int(lm.y * img_h) for lm in landmarks)
            face_coords = (x_min, y_min, x_max - x_min, y_max - y_min) # (x, y, w, h)

            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
            def get_coords(idxs):
                coords = []
                for i in idxs:
                    lm = landmarks[i]
                    coords.append((int(lm.x * img_w), int(lm.y * img_h)))
                return np.array(coords)

            # 1. EAR DURATION CHECK
            left_ear = _eye_aspect_ratio(get_coords(L_EYE_IDXS))
            right_ear = _eye_aspect_ratio(get_coords(R_EYE_IDXS))
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < EAR_THRESHOLD:
                if self.EAR_CLOSED_START_TIME is None:
                    self.EAR_CLOSED_START_TIME = current_time
                
                if (current_time - self.EAR_CLOSED_START_TIME) >= EAR_DURATION_ALERT_SEC and not self.EAR_ALERT_CONFIRMED:
                    self.EAR_ALERT_COUNT += 1
                    self.EAR_ALERT_CONFIRMED = True
                    ear_alert_instant = True
            else:
                self.EAR_CLOSED_START_TIME = None
                self.EAR_ALERT_CONFIRMED = False 

            # 2. MAR DURATION CHECK
            mouth_coords = get_coords(MOUTH_IDXS)
            mar = _mouth_aspect_ratio(mouth_coords)

            if mar > MAR_THRESHOLD:
                if self.MAR_YAWN_START_TIME is None:
                    self.MAR_YAWN_START_TIME = current_time

                if (current_time - self.MAR_YAWN_START_TIME) >= MAR_DURATION_ALERT_SEC and not self.MAR_ALERT_CONFIRMED:
                    self.MAR_ALERT_COUNT += 1
                    self.MAR_ALERT_CONFIRMED = True
                    mar_alert_instant = True
            else:
                self.MAR_YAWN_START_TIME = None
                self.MAR_ALERT_CONFIRMED = False 

            # Draw indicators (colors change on instant alert)
            cv2.polylines(image, [get_coords(L_EYE_IDXS)], True, (255, 0, 255) if ear_alert_instant else (0, 255, 0), 2)
            cv2.polylines(image, [get_coords(R_EYE_IDXS)], True, (255, 0, 255) if ear_alert_instant else (0, 255, 0), 2)
            cv2.polylines(image, [mouth_coords], True, (0, 0, 255) if mar_alert_instant else (0, 255, 255), 2)

            # 3. DECIDE SLEEPINESS LEVEL AND APPLY RECOVERY
            new_level = self._update_sleepiness_level()
            self._apply_recovery_logic(new_level, avg_ear, mar)
            self.SLEEPINESS_LEVEL = self._update_sleepiness_level() 

            # --- Terminal Output ---
            self._print_terminal_alert(self.SLEEPINESS_LEVEL, avg_ear, mar)
            
        else:
            # No face detected - reset all timers
            self.EAR_CLOSED_START_TIME = None
            self.MAR_YAWN_START_TIME = None
            self.EAR_ALERT_CONFIRMED = False
            self.MAR_ALERT_CONFIRMED = False
            
            if self.recovery_countdown_active:
                self.recovery_countdown_active = False 
            self.alert_start_time = current_time 

            cv2.putText(image, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if current_time - self.last_terminal_output >= TERMINAL_OUTPUT_INTERVAL:
                print(f"{Colors.RED}[{datetime.now().strftime('%H:%M:%S')}] ❌ No face detected{Colors.END}")
                self.last_terminal_output = current_time

        return image, face_detected, face_coords


# ==============================================================================
# --- EMOTION DETECTION COMPONENTS (Load, Logic) ---
# ==============================================================================

# Setup (Load models outside the loop)
try:
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Assuming 'Models/emotion_model.h5' is the correct path structure
    classifier = load_model(os.path.join('Models', 'emotion_model.h5')) 
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
except Exception as e:
    print(f"{Colors.RED}ERROR: Could not load Emotion Detection models. Emotion feature will be skipped. Error: {e}{Colors.END}")
    face_classifier = None
    classifier = None

def run_emotion_detection(frame, face_coords_drowsiness=None):
    """
    Runs emotion detection on the frame. If face_coords_drowsiness is provided, 
    it uses that bounding box instead of running a full detectMultiScale.
    """
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if face_classifier is None or classifier is None:
        # Skip if models failed to load
        return frame
    
    faces = []
    
    if face_coords_drowsiness:
        # Use the face found by the Drowsiness Detector (more robust MediaPipe detection)
        (x, y, w, h) = face_coords_drowsiness
        faces.append((x, y, w, h))
    else:
        # Fallback to Haar cascade if MediaPipe failed or wasn't used
        faces = face_classifier.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4, 
            minSize=(40, 40)
        )

    if len(faces) > 0:
        # Sort faces by area (w * h) in descending order and use the largest
        if not face_coords_drowsiness:
             faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
             
        (x, y, w, h) = faces[0]
        
        # Draw bounding box on the face for Emotion (Green/Yellow to differentiate Drowsiness mesh)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2) # Orange box

        # Prepare ROI for classification, handle potential out-of-bounds
        roi_gray = gray[max(0, y):y + h, max(0, x):x + w]
        
        if roi_gray.size == 0:
            return frame # Skip if ROI is invalid
            
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            
            roi = np.expand_dims(roi, axis=0)  
            roi = np.expand_dims(roi, axis=-1) 

            prediction = classifier.predict(roi, verbose=0)[0] 
            label = emotion_labels[prediction.argmax()]
            
            # Display emotion label slightly above the box
            label_position = (x, y - 10) 
            cv2.putText(frame, f"Emotion: {label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
    return frame

# ==============================================================================
# --- MAIN APPLICATION ENTRY POINT ---
# ==============================================================================

def start_integrated_detection():
    """Initializes and runs the integrated video processing loop for Drowsiness and Emotion."""
    
    # 1. Initialization
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)

    # 2. Startup Messages
    print(f"{Colors.BOLD}{Colors.BLUE}🚀 Starting Integrated Drowsiness + Emotion Detector 🚀{Colors.END}")
    print(f"{Colors.GREEN}Press 'q' to quit the application{Colors.END}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.END}")
    
    if not cap.isOpened():
        print(f"{Colors.RED}ERROR: Could not open video stream (Webcam index 0).{Colors.END}")
        return

    # 3. Main Video Processing Loop
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        try:
            image = cv2.flip(image, 1) # Mirror the image
            
            # --- 3a. Drowsiness Detection ---
            # Returns: annotated image, whether a face was found, and the face's bounding box
            processed_image, face_detected, face_coords = detector._run_drowsiness_logic(image)
            
            # --- 3b. Emotion Detection ---
            # Reuse the image and the face coordinates found by the Drowsiness Detector
            processed_image = run_emotion_detection(processed_image, face_coords)
            
            # --- 3c. Final Drowsiness Status Display (Drawn over everything) ---
            img_h, img_w, _ = processed_image.shape
            cv2.rectangle(processed_image, (0, 0), (img_w, 80), (0, 0, 0), -1) # Clear top bar
            detector._draw_display_status(processed_image, img_w, 0, 0) # Metrics args are placeholders here

            cv2.imshow('Integrated Driver Monitoring (Drowsiness & Emotion)', processed_image)
            
        except Exception as e:
            print(f"{Colors.RED}An error occurred during frame processing: {e}{Colors.END}")
            # Optionally, break or continue based on error severity
            pass

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print(f"{Colors.GREEN}👋 Application terminated by user{Colors.END}")
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"{Colors.BLUE}🔚 Integrated detector stopped{Colors.END}")

if __name__ == '__main__':
    start_integrated_detection()