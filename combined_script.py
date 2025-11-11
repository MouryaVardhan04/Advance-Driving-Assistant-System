import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from datetime import datetime as dt 
import os 
# import pyttsx3 # REMOVED: No more text-to-speech

# *** NEW MP3 LIBRARIES ***
from pydub import AudioSegment
import simpleaudio as sa
# *************************

import eel 
import requests 

# ====================================================================
# 🔥 CRITICAL: Removed GEMINI API Key and URL constants! 🔥
# ====================================================================

# --- GLOBAL STATE ---
CURRENT_PLAYBACK = None 
PRELOADED_AUDIO = None # Global to store decoded audio
# --------------------

# --- CONFIGURATION CONSTANTS ---
EAR_THRESHOLD = 0.25      
MAR_THRESHOLD = 0.6       
TERMINAL_OUTPUT_INTERVAL = 2  
# *** MODIFIED FOR INSTANT RESPONSE (0.5 seconds) ***
EAR_DURATION_ALERT_SEC = 1.5
MAR_DURATION_ALERT_SEC = 4.0 
RECOVERY_TIME_SEC = 20.0 
EAR_ALERT_LIMIT_L1 = 3  
MAR_ALERT_LIMIT_L1 = 2  
EAR_ALERT_LIMIT_L2 = 5  
MAR_ALERT_LIMIT_L2 = 3  

# --- AUDIO ALARM CONFIGURATION (Now supports MP3) ---
ALARM_SOUND_FILE = 'beep.mp3' 
# -------------------------------------

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

# --- NEW: AUDIO PRELOAD FUNCTION ---
def preload_audio(filename):
    """Loads and decodes the audio file once at startup."""
    global PRELOADED_AUDIO
    try:
        if not os.path.exists(filename):
            print(f"{Colors.RED}[AUDIO ALARM] Preload Error: File not found at {filename}.{Colors.END}")
            return False
        
        # Load and decode the audio file once at startup (Expensive operation done only once)
        PRELOADED_AUDIO = AudioSegment.from_file(filename)
        print(f"[AUDIO ALARM] Successfully preloaded {filename}.")
        return True
    except Exception as e:
        # Check if error is related to FFmpeg
        if "No such file or directory" in str(e) and ("ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower()):
             print(f"{Colors.RED}[AUDIO ALARM] Preload FAILED: FFmpeg command not found. Install FFmpeg.{Colors.END}")
        else:
             print(f"{Colors.RED}[AUDIO ALARM] Preload FAILED: Check FFmpeg installation/permissions. Error: {e}{Colors.END}")
        return False


# --- UPDATED: AUDIO UTILITY FUNCTION (MP3-Compatible, Uses Preloaded Data) ---
def play_sound_alarm():
    """Plays the pre-loaded MP3 audio buffer, non-blocking."""
    global CURRENT_PLAYBACK, PRELOADED_AUDIO
    
    if PRELOADED_AUDIO is None:
        return

    # Stop any currently playing sound before starting a new one
    stop_sound_alarm()
    
    try:
        audio = PRELOADED_AUDIO
        
        # Play the audio buffer (Fast operation)
        CURRENT_PLAYBACK = sa.play_buffer(
            audio.raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )
        
    except Exception as e:
        print(f"{Colors.RED}[AUDIO ALARM] Failed to play sound: {e}{Colors.END}")


def stop_sound_alarm():
    """Stops the currently playing sound (if any)."""
    global CURRENT_PLAYBACK
    if CURRENT_PLAYBACK is not None and CURRENT_PLAYBACK.is_playing():
        CURRENT_PLAYBACK.stop()
        # Removed print for slight performance gain

# --- END AUDIO UTILITY FUNCTION ---

def safe_eel_call(func_name, *args):
    """Safely call Eel functions, with fallback if not available"""
    if func_name == "DisplayMessage":
        print(f"[Eel] DisplayMessage: {args[0]}") 
    elif func_name == "setMicState":
        print(f"[Eel] Setting Mic State to: {args[0]}")
    elif func_name == "showHood":
        print("[Eel] Calling showHood")
    else:
        print(f"[Eel] Calling {func_name}")

# --- HELPER FUNCTIONS (Kept) ---
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

# --- DROWSINESS DETECTOR CLASS ---
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

        if ear_count > EAR_ALERT_LIMIT_L2 or mar_count > MAR_ALERT_LIMIT_L2:
            return 3
        elif ear_count >= EAR_ALERT_LIMIT_L2 or mar_count >= MAR_ALERT_LIMIT_L2:
            return 2
        elif ear_count >= EAR_ALERT_LIMIT_L1 or mar_count >= MAR_ALERT_LIMIT_L1:
            return 1
        else:
            return 0 

    def _apply_recovery_logic(self, current_level, avg_ear, mar):
        """Decrements EAR and MAR counts by 1 after 20 seconds of continuous alertness."""
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
                
                print(f"{Colors.GREEN}[{dt.now().strftime('%H:%M:%S')}] 🧠 RECOVERY: Alertness held for {RECOVERY_TIME_SEC}s. Counts decremented by 1. New Counts (E:{self.EAR_ALERT_COUNT}, M:{self.MAR_ALERT_COUNT}){Colors.END}")

                self.alert_start_time = current_time 
        else:
            self.recovery_countdown_active = False 
            self.alert_start_time = current_time 

    def _print_terminal_alert(self, level, ear, mar):
        """Prints simplified, colored terminal output for drowsiness alerts."""
        timestamp = dt.now().strftime("%H:%M:%S")
        
        if level > 0 or time.time() - self.last_terminal_output >= TERMINAL_OUTPUT_INTERVAL:
            common_stats = f" | Counts (EAR: {self.EAR_ALERT_COUNT}, MAR: {self.MAR_ALERT_COUNT}) | Metrics (EAR: {ear:.3f}, MAR: {mar:.3f})"
            
            if level == 3:
                print(f"{Colors.RED}{Colors.BOLD}[{timestamp}] 🚨 DEEP SLEEP DETECTED! (Level 3) 🚨{common_stats}{Colors.END}")
            elif level == 2:
                print(f"{Colors.YELLOW}{Colors.BOLD}[{timestamp}] ⚠️  MEDIUM SLEEP DETECTED! (Level 2) ⚠️{common_stats}{Colors.END}")
            elif level == 1:
                print(f"{Colors.CYAN}[{timestamp}] 😴 NORMAL SLEEP DETECTED! (Level 1){common_stats}{Colors.END}")
            else:
                if self.recovery_countdown_active:
                    time_left = max(0, RECOVERY_TIME_SEC - (time.time() - self.alert_start_time))
                    print(f"{Colors.GREEN}[{timestamp}] ✅ Alert - Level 0 (Recovery Time Left: {time_left:.0f}s){common_stats}{Colors.END}")
                else:
                    print(f"{Colors.GREEN}[{timestamp}] ✅ Fully Alert{common_stats}{Colors.END}")

            self.last_terminal_output = time.time()

    def _draw_display_status(self, image, img_w):
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
        
        # Keeping the visual text display on the frame for level information
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(image, f"EAR Count: {self.EAR_ALERT_COUNT}", (img_w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"MAR Count: {self.MAR_ALERT_COUNT}", (img_w - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image

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
                # EYES ARE CLOSED (OR NEARLY CLOSED)
                if self.EAR_CLOSED_START_TIME is None:
                    self.EAR_CLOSED_START_TIME = current_time
                
                # Check for DURATION and trigger alarm/increment count
                if (current_time - self.EAR_CLOSED_START_TIME) >= EAR_DURATION_ALERT_SEC:
                    if not self.EAR_ALERT_CONFIRMED:
                        self.EAR_ALERT_COUNT += 1
                        self.EAR_ALERT_CONFIRMED = True
                        ear_alert_instant = True
                        # Immediately play the beep sound here!
                        play_sound_alarm() # Uses preloaded audio
                    elif self.SLEEPINESS_LEVEL >= 1:
                        # Re-trigger beep on subsequent frames if currently drowsy (Level 1+)
                        play_sound_alarm() # Uses preloaded audio
            else:
                # EYES ARE OPEN (ABOVE THRESHOLD)
                if self.EAR_CLOSED_START_TIME is not None:
                    # *** MODIFICATION: Stop beep immediately when eyes open ***
                    stop_sound_alarm()
                
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
            # No face detected - reset all timers and stop sound
            self.EAR_CLOSED_START_TIME = None
            self.MAR_YAWN_START_TIME = None
            self.EAR_ALERT_CONFIRMED = False
            self.MAR_ALERT_CONFIRMED = False
            stop_sound_alarm() # Stop any current beeping
            
            if self.recovery_countdown_active:
                self.recovery_countdown_active = False 
            self.alert_start_time = current_time 

            cv2.putText(image, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if current_time - self.last_terminal_output >= TERMINAL_OUTPUT_INTERVAL:
                print(f"{Colors.RED}[{dt.now().strftime('%H:%M:%S')}] ❌ No face detected{Colors.END}")
                self.last_terminal_output = current_time

        return image, face_detected, face_coords 

# --- DUMMY EEL FUNCTIONS (Kept) ---
@eel.expose 
def dummy_takecommand():
    """Dummy function to satisfy any frontend dependencies that might call takecommand."""
    return "unrecognized"

@eel.expose
def dummy_start_conversation():
    """Dummy function for starting conversation (now a no-op)."""
    print("[Eel] Conversation feature removed. Ignoring start_conversation call.")
    safe_eel_call("setMicState", "idle") 
    
@eel.expose
def dummy_allCommands(query=""):
    """Dummy function for text commands (now a no-op)."""
    print(f"[Eel] Conversation feature removed. Ignoring command: {query}")
    safe_eel_call("showHood") 

# --- ALARM MANAGEMENT (Simplified) ---

def handle_drowsiness_alarm(current_level, last_level):
    """Handles alarm management (now primarily stopping the audio on recovery)."""
    
    # 1. Deactivation/Recovery - This acts as a fallback for stopping sound
    if current_level == 0 and last_level > 0:
        stop_sound_alarm() 
        safe_eel_call("DisplayMessage", "Alertness Recovered. Monitoring silently.")

    return 

# --- MAIN APPLICATION ENTRY POINT ---

def start_integrated_detection():
    """Initializes and runs the integrated video processing loop for Drowsiness only."""
    
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)

    # *** NEW: PRELOAD AUDIO HERE ***
    if not preload_audio(ALARM_SOUND_FILE):
        print(f"{Colors.RED}FATAL: Audio alarm will not work. Check setup.{Colors.END}")

    print(f"{Colors.BOLD}{Colors.BLUE}🚀 Starting Drowsiness Detector (MP3 Beep Alarm) 🚀{Colors.END}")
    print(f"{Colors.RED}*** REQUIRES FFmpeg AND 'simpleaudio' installed! ***{Colors.END}")
    print(f"{Colors.GREEN}Audio Alarm File: {ALARM_SOUND_FILE}{Colors.END}")
    print(f"{Colors.GREEN}Alarm triggers after {EAR_DURATION_ALERT_SEC} seconds of eye closure.{Colors.END}")
    print(f"{Colors.GREEN}Press 'q' to quit the application{Colors.END}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.END}")
    
    if not cap.isOpened():
        print(f"{Colors.RED}ERROR: Could not open video stream (Webcam index 0).{Colors.END}")
        return
    
    last_drowsiness_level = 0 # Track the level to detect escalation
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        try:
            image = cv2.flip(image, 1) 
            
            # --- 1. Drowsiness Detection & Metrics (Now includes beep logic) ---
            processed_image, face_detected, face_coords = detector._run_drowsiness_logic(image)
            current_level = detector.SLEEPINESS_LEVEL
            
            # --- 2. ALARM MANAGEMENT LOGIC (Recovery only) ---
            handle_drowsiness_alarm(current_level, last_drowsiness_level)
            last_drowsiness_level = current_level

            # --- 3. Final Status Display ---
            img_h, img_w, _ = processed_image.shape
            cv2.rectangle(processed_image, (0, 0), (img_w, 80), (0, 0, 0), -1) 
            detector._draw_display_status(processed_image, img_w)

            cv2.imshow('Driver Monitoring (Beep Alarm Only)', processed_image)
            
        except Exception as e:
            print(f"{Colors.RED}An error occurred during frame processing: {e}{Colors.END}")
            # Ensure sound stops on crash
            stop_sound_alarm()
            pass

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print(f"{Colors.GREEN}👋 Application terminated by user{Colors.END}")
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stop_sound_alarm() # Stop any active sound
    print(f"{Colors.BLUE}🔚 Integrated detector stopped{Colors.END}")

if __name__ == '__main__':
    start_integrated_detection()