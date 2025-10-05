# drowsiness.py

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from datetime import datetime

# --- Configuration Constants ---
EAR_THRESHOLD = 0.25      # Eye Aspect Ratio threshold (Eye Closure)
CONSEC_FRAMES_EAR = 48    # Frames for EAR alert (~1.6 seconds at 30 FPS)

MAR_THRESHOLD = 0.6       # Mouth Aspect Ratio threshold (Yawning)
CONSEC_FRAMES_MAR = 15    # Frames for MAR alert (~0.5 seconds)

TERMINAL_OUTPUT_INTERVAL = 2  # Seconds for terminal output throttling

# Landmark Indices for Calculations
R_EYE_IDXS = [33, 160, 158, 133, 153, 144] 
L_EYE_IDXS = [362, 385, 387, 263, 373, 380] 
MOUTH_IDXS = [61, 291, 0, 17, 14, 37, 267, 40, 270, 310, 317, 82, 312]

# Terminal output colors
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

# --- Helper Functions (Private to the module) ---

def _eye_aspect_ratio(eye):
    # Vertical distances (P2-P6 and P3-P5)
    A = dist.euclidean(eye[1], eye[5]) 
    B = dist.euclidean(eye[2], eye[4]) 
    # Horizontal distance (P1-P4)
    C = dist.euclidean(eye[0], eye[3]) 
    ear = (A + B) / (2.0 * C)
    return ear

def _mouth_aspect_ratio(mouth):
    # Vertical distances (A, B, C)
    A = dist.euclidean(mouth[1], mouth[11]) 
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[3], mouth[9])
    # Horizontal distance (D)
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
        
        # Counters and Status
        self.EAR_COUNTER = 0
        self.MAR_COUNTER = 0
        self.SLEEPINESS_LEVEL = 0
        self.last_terminal_output = 0

    def _update_sleepiness_level(self, ear_alert, mar_alert):
        """Combines EAR and MAR alerts into a cumulative sleepiness score."""
        score = 0
        
        if ear_alert:
            score += 2 # Eye closure is the strongest indicator
        if mar_alert:
            score += 1

        if score == 3:
            return 3 # Critical Drowsiness (Eyes Closed + Yawning)
        elif score == 2:
            return 2 # Medium Drowsiness (Eyes Closed only)
        elif score == 1:
            return 1 # Low Drowsiness (Yawning only)
        else:
            return 0 # Alert

    def _print_terminal_alert(self, level, ear, mar):
        """Prints simplified, colored terminal output for drowsiness alerts."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Throttling Logic: Only print detailed alerts or 'Alert' status after the interval
        if level > 0 or time.time() - self.last_terminal_output >= TERMINAL_OUTPUT_INTERVAL:
            
            if level == 3:
                print(f"{Colors.RED}{Colors.BOLD}[{timestamp}] 🚨 CRITICAL DROWSINESS DETECTED! 🚨{Colors.END}")
                print(f"{Colors.RED}   Level: HIGH (3) - Eyes: {ear:.3f}, Mouth: {mar:.3f}{Colors.END}")
                print(f"{Colors.RED}   ⚠️  IMMEDIATE ATTENTION REQUIRED! ⚠️{Colors.END}")
            elif level == 2:
                print(f"{Colors.YELLOW}{Colors.BOLD}[{timestamp}] ⚠️  DROWSY DETECTED! ⚠️{Colors.END}")
                print(f"{Colors.YELLOW}   Level: MEDIUM (2) - Eyes: {ear:.3f}, Mouth: {mar:.3f}{Colors.END}")
                print(f"{Colors.YELLOW}   💤 Consider taking a break! 💤{Colors.END}")
            elif level == 1:
                print(f"{Colors.CYAN}[{timestamp}] ⚡ FATIGUE WARNING ⚡{Colors.END}")
                print(f"{Colors.CYAN}   Level: LOW (1) - Eyes: {ear:.3f}, Mouth: {mar:.3f}{Colors.END}")
            else:
                # 'Alert' message output
                print(f"{Colors.GREEN}[{timestamp}] ✅ Alert - Eyes: {ear:.3f}, Mouth: {mar:.3f}{Colors.END}")
            
            self.last_terminal_output = time.time()


    def process_frame(self, image):
        """
        Processes a single video frame to detect EAR and MAR.
        Returns the annotated image, sleepiness level, and a success flag.
        """
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        image.flags.writeable = True
        
        img_h, img_w, _ = image.shape
        ear_alert, mar_alert = False, False
        avg_ear, mar = 0, 0
        
        # Reset alert display area
        cv2.rectangle(image, (0, 0), (img_w, 80), (0, 0, 0), -1) 

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Draw complete face mesh
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
            # Helper to convert normalized landmark to pixel coordinates
            def get_coords(idxs):
                coords = []
                for i in idxs:
                    lm = landmarks[i]
                    coords.append((int(lm.x * img_w), int(lm.y * img_h)))
                return np.array(coords)

            # 1. EYE ASPECT RATIO (EAR)
            left_ear = _eye_aspect_ratio(get_coords(L_EYE_IDXS))
            right_ear = _eye_aspect_ratio(get_coords(R_EYE_IDXS))
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < EAR_THRESHOLD:
                self.EAR_COUNTER += 1
                if self.EAR_COUNTER >= CONSEC_FRAMES_EAR:
                    ear_alert = True
            else:
                self.EAR_COUNTER = 0

            # 2. MOUTH ASPECT RATIO (MAR)
            mouth_coords = get_coords(MOUTH_IDXS)
            mar = _mouth_aspect_ratio(mouth_coords)

            if mar > MAR_THRESHOLD:
                self.MAR_COUNTER += 1
                if self.MAR_COUNTER >= CONSEC_FRAMES_MAR:
                    mar_alert = True
            else:
                self.MAR_COUNTER = 0

            # Draw indicators (colors change on alert)
            cv2.polylines(image, [get_coords(L_EYE_IDXS)], True, (255, 0, 255) if ear_alert else (0, 255, 0), 2)
            cv2.polylines(image, [get_coords(R_EYE_IDXS)], True, (255, 0, 255) if ear_alert else (0, 255, 0), 2)
            cv2.polylines(image, [mouth_coords], True, (0, 0, 255) if mar_alert else (0, 255, 255), 2)

            # 3. DECIDE SLEEPINESS LEVEL
            self.SLEEPINESS_LEVEL = self._update_sleepiness_level(ear_alert, mar_alert)

            # --- Display Logic ---
            status_color = (0, 255, 0)
            status_text = "ALERT"
            
            if self.SLEEPINESS_LEVEL == 3:
                status_text = "CRITICAL DROWSINESS (EAR+MAR)"
                status_color = (0, 0, 255) # RED
            elif self.SLEEPINESS_LEVEL == 2:
                status_text = "DROWSY - EYES CLOSED"
                status_color = (0, 165, 255) # ORANGE
            elif self.SLEEPINESS_LEVEL == 1:
                status_text = "FATIGUE - YAWN DETECTED"
                status_color = (0, 255, 255) # YELLOW
            
            # Draw Status
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Draw Metrics
            cv2.putText(image, "EAR: {:.2f}".format(avg_ear), (img_w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, "MAR: {:.2f}".format(mar), (img_w - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw Counters
            cv2.putText(image, f"EAR Counter: {self.EAR_COUNTER}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"MAR Counter: {self.MAR_COUNTER}", (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Terminal Output
            self._print_terminal_alert(self.SLEEPINESS_LEVEL, avg_ear, mar)
            
        else:
            # No face detected
            cv2.putText(image, "NO FACE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Throttle "No face detected" message
            if time.time() - self.last_terminal_output >= TERMINAL_OUTPUT_INTERVAL:
                print(f"{Colors.RED}[{datetime.now().strftime('%H:%M:%S')}] ❌ No face detected{Colors.END}")
                self.last_terminal_output = time.time()
                
        return image, self.SLEEPINESS_LEVEL, True # Success (frame processed)

# --- Primary Application Entry Point ---

def start_application():
    """Initializes and runs the main video processing loop."""
    
    # 1. Initialization
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)

    # 2. Startup Messages
    print(f"{Colors.BOLD}{Colors.BLUE}🚀 Starting Simplified Drowsiness Detector (EAR + MAR) 🚀{Colors.END}")
    print(f"{Colors.CYAN}EAR/MAR Thresholds: {EAR_THRESHOLD} / {MAR_THRESHOLD}{Colors.END}")
    print(f"{Colors.GREEN}Press 'q' to quit the application{Colors.END}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.END}")
    
    if not cap.isOpened():
        print(f"{Colors.RED}ERROR: Could not open video stream or file (Webcam index 0).{Colors.END}")
        return

    # 3. Main Video Processing Loop
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        try:
            # Flip the image horizontally for a more natural selfie-view
            image = cv2.flip(image, 1)
            
            # Process the frame using the detector object
            processed_image, sleepiness_level, processed = detector.process_frame(image)

            # Display the resulting frame
            cv2.imshow('Simplified Drowsiness Detector', processed_image)
        except Exception as e:
            # Catch exceptions during frame processing (e.g., if MediaPipe fails)
            print(f"{Colors.RED}An error occurred during frame processing: {e}{Colors.END}")
            # Do not break here to allow recovery, but you could if errors are persistent
            pass

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print(f"{Colors.GREEN}👋 Application terminated by user{Colors.END}")
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"{Colors.BLUE}🔚 Drowsiness detector stopped{Colors.END}")