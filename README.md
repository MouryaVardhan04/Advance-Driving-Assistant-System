# Advanced Driving Assistant System (ADAS)

A comprehensive real-time driver monitoring and road safety system that combines multiple computer vision modules to assist drivers and prevent accidents. The system integrates drowsiness detection, lane line detection, road sign recognition, pothole detection, and pedestrian detection into a unified ADAS platform.

## Problem Statement & Real-time Need
- Driver inattention, drowsiness and road hazards cause accidents
- Existing systems are either expensive or offline/slow
- Need for a compact, low-latency on-device solution
- Detect and alert in real-time to reduce incidents

## How Our Project Resolves the Problem
- Multi-module pipeline: drowsiness, road-signs, potholes, pedestrians, lanes
- On-device processing with MediaPipe + OpenCV for low latency
- Immediate audio alarm on danger and visual overlays for driver
- Modular design — each module runs independently and can be tuned

## 🎯 Project Overview

This ADAS project implements a multi-module system designed for real-time driver monitoring and road safety assistance. The system processes video streams from forward-facing and driver-facing cameras to provide immediate alerts and visual overlays, helping drivers maintain alertness and awareness of road conditions.

### Key Features
- **Real-time Processing**: Low-latency detection using optimized computer vision algorithms
- **Multi-Module Integration**: Unified pipeline combining 5 distinct detection modules
- **Audio Alerts**: Immediate beep alarms for critical events (drowsiness, hazards)
- **Visual Overlays**: Informative HUD-style displays with lane curvature, vehicle position, and detected objects
- **Modular Architecture**: Each module operates independently and can be easily extended or modified
- **Lightweight Design**: OpenCV + MediaPipe-based modules optimized for in-vehicle deployment
- **Low-light Capability**: Adaptive preprocessing for robust detection in dim lighting conditions

---

## 📦 Modules Overview

The system consists of five main modules:

1. **Drowsiness Detection** - Monitors driver alertness using eye and mouth tracking
2. **Lane Line Detection** - Detects and tracks lane markings, calculates curvature and vehicle position
3. **Road Sign Detection** - Recognizes traffic signs and displays appropriate warnings
4. **Pothole Detection** - Identifies road surface defects using deep learning
5. **Pedestrian Detection** - Detects pedestrians in the vehicle's path

---

## 🔧 Module 1: Drowsiness Detection

### Architecture
The drowsiness detection module uses **MediaPipe Face Mesh** to extract facial landmarks and calculates physiological metrics to determine driver alertness.

**Processing Pipeline:**
```
Video Frame → MediaPipe Face Mesh → Landmark Extraction → 
EAR/MAR Calculation → Temporal Analysis → Alarm Trigger
```

### Technical Details

#### Eye Aspect Ratio (EAR)
- **Formula**: `EAR = (A + B) / (2.0 * C)`
  - A = vertical distance between eye landmarks 1-5
  - B = vertical distance between eye landmarks 2-4
  - C = horizontal distance between eye landmarks 0-3
- **Threshold**: `EAR_THRESHOLD = 0.25`
- **Alert Duration**: 1.0 second of eye closure triggers alarm

#### Mouth Aspect Ratio (MAR)
- **Formula**: `MAR = (A + B + C) / (3.0 * D)`
  - Measures mouth openness to detect yawning
- **Threshold**: `MAR_THRESHOLD = 0.6`
- **Alert Duration**: 2.0 seconds triggers alert

#### Sleepiness Levels
The system implements a 4-level alert system:
- **Level 0**: Fully alert (EAR ≥ 0.25, MAR ≤ 0.6)
- **Level 1**: Normal sleepiness (3 EAR alerts OR 2 MAR alerts)
- **Level 2**: Medium sleepiness (≥ 5 EAR alerts OR ≥ 3 MAR alerts)
- **Level 3**: Deep sleep detected (severe drowsiness)

#### Recovery Mechanism
- After 20 seconds of continuous alertness, alert counts decrement by 1
- Prevents false positives from temporary eye closures

### Audio Alarm System
- **Pre-loaded Audio**: MP3 file is decoded once at startup using `pydub` and `simpleaudio`
- **Ultra-low Latency**: Direct audio buffer playback eliminates file I/O delays
- **Immediate Response**: Alarm starts within milliseconds of detection
- **Auto-stop**: Sound stops immediately when eyes reopen
- **Minimal UI**: Clean on-frame display with essential information only

### Low-light Performance
- **Adaptive Preprocessing**: Robust landmark tracking improves detection in dim scenes
- **Enhanced Robustness**: MediaPipe Face Mesh maintains accuracy under varying lighting conditions
- **Real-time Operation**: Operates effectively in low-light conditions without significant performance degradation

### Libraries Used
- `mediapipe` - Face mesh landmark detection
- `scipy.spatial.distance` - Euclidean distance calculations for EAR/MAR
- `pydub` - Audio file decoding
- `simpleaudio` - Low-latency audio playback
- `opencv-python` - Video capture and frame processing
- `numpy` - Numerical computations

### Key Files
- `combined_script.py` - Main drowsiness detection implementation
- `beep.mp3` - Alarm audio file

---

## 🔧 Module 2: Lane Line Detection

### Architecture
Lane detection uses a classical computer vision pipeline combining camera calibration, perspective transformation, and polynomial fitting.

**Processing Pipeline:**
```
Video Frame → Camera Undistortion → Perspective Transform (Bird's Eye) → 
Color/Gradient Thresholding → Sliding Window Search → Polynomial Fitting → 
Curvature Calculation → Overlay Rendering
```

### Technical Details

#### Camera Calibration
- Uses chessboard pattern images to correct lens distortion
- Calculates camera matrix and distortion coefficients
- Input: 9x6 chessboard images from `camera_cal/` directory
- Output: Undistorted frames with corrected geometry

#### Perspective Transformation
- Transforms front view to bird's-eye view for easier lane detection
- **Source Points** (front view):
  - Top-left: (600, 460)
  - Bottom-left: (200, 720)
  - Bottom-right: (1080, 720)
  - Top-right: (680, 460)
- **Destination Points** (bird's-eye):
  - Transforms to rectangular view (200, 0) to (1000, 720)

#### Thresholding
Combines multiple color space thresholds:
- **HLS Color Space**: L-channel for right lane (white/yellow)
  - Threshold: 80-100% of value range
  - Applied to right half of image (x > 800)
- **HSV Color Space**: H-channel and V-channel for left lane (yellow)
  - H-channel: 20-30 (yellow hue range)
  - V-channel: 70-100% of value range
  - Applied to left half of image (x < 480)

#### Sliding Window Search
- Divides image into 9 horizontal windows
- Window margin: 80 pixels
- Minimum pixels per window: 50
- Tracks left and right lane centers across windows

#### Polynomial Fitting
- Fits 2nd-degree polynomial: `x = ay² + by + c`
- Minimum lane pixels required: 1000 per side
- Validates lane detection based on:
  - Minimum pixels: 1200 per lane
  - Lane width: 200-1000 pixels at bottom of image

#### Curvature Calculation
- **Meters per pixel**:
  - `ym_per_pix = 30/720` (vertical)
  - `xm_per_pix = 3.7/700` (horizontal, US standard lane width)
- **Radius of Curvature**: `R = (1 + (2ay + b)²)^(3/2) / |2a|`
- Evaluated at y = 700 pixels (bottom of image)

#### Vehicle Position
- Calculates distance from lane center
- Formula: `position = (image_center - lane_center) × xm_per_pix`

### Direction Detection
- Analyzes polynomial coefficients to determine curve direction
- **Straight**: |coefficient| ≤ 0.00015
- **Left Curve**: coefficient < 0
- **Right Curve**: coefficient > 0
- Uses 10-frame rolling window for stability

### Key Features
- **Novelty Feature**: Edge and Hough-based lane boundary detection with perspective transform
- **Drift Warnings**: Provides lane overlay + drift warnings
- **Scene Context Fusion**: Works with road-sign & pothole modules to fuse scene context
- **Runtime Optimization**: Optimized to keep frame processing fast

### Visual Overlay
The system displays:
- Direction icon (left turn, right turn, straight)
- Curvature radius in meters
- Vehicle position relative to lane center
- "Good Lane Keeping" status indicator

### Libraries Used
- `opencv-python` - Image processing and camera calibration
- `numpy` - Polynomial fitting and mathematical operations
- `matplotlib` - Image loading for direction icons

### Key Files
- `LaneLines.py` - Core lane detection class
- `CameraCalibration.py` - Camera calibration utilities
- `Thresholding.py` - Color space thresholding
- `PerspectiveTransformation.py` - Perspective transformation
- `left_turn.png`, `right_turn.png`, `straight.png` - Direction icons

---

## 🔧 Module 3: Road Sign Detection

### Architecture
Road sign detection uses a pre-trained convolutional neural network (CNN) to classify traffic signs in real-time.

**Processing Pipeline:**
```
Video Frame → Resize (32×32) → Grayscale → Histogram Equalization → 
Normalization → CNN Inference → Confidence Threshold (75%) → Icon Overlay
```

### Technical Details

#### Model Architecture
- **Input Size**: 32×32 grayscale images
- **Framework**: TensorFlow/Keras (`.h5` format)
- **Model File**: `Models/sign_model.h5`
- **Preprocessing**:
  1. Resize input frame to 32×32
  2. Convert to grayscale
  3. Apply histogram equalization (adaptive contrast)
  4. Normalize to [0, 1] range
  5. Reshape to (1, 32, 32, 1) for CNN input

#### Detection Logic
- Processes entire frame (single forward pass)
- Returns top prediction with confidence score
- **Display Threshold**: Only displays if confidence ≥ 75%
- When threshold met, shows:
  - Sign icon from `asserts/` directory (class_index.png)
  - Sign name label
  - Semi-transparent widget in top-right corner

#### Label Mapping
- Labels loaded from `labels.csv`
- Format: `ClassId, Name`
- Maps class index to human-readable sign names
- Default fallback: Speed Limit, Stop, Yield

### Visual Display
- **Widget Position**: Top-right corner (margin: 20px)
- **Widget Size**: 220×160 pixels
- **Icon Size**: Auto-scaled to fit (max 180×100)
- **Background**: Semi-transparent dark box (alpha: 0.55)
- **Label**: Centered at bottom of widget

### Libraries Used
- `tensorflow` / `keras` - CNN model loading and inference
- `opencv-python` - Image preprocessing
- `pandas` - CSV label file parsing
- `numpy` - Array operations

### Key Features
- **Lightweight Inference**: Near real-time performance with optimized CNN architecture
- **Extensible Design**: Can be extended with geo-fencing or sign-specific rules
- **Confidence-based Display**: Only shows signs above 75% confidence to reduce false positives

### Key Files
- `sign_lane.py` - Contains `RoadSignDetector` class
- `Models/sign_model.h5` - Pre-trained CNN model
- `labels.csv` - Class ID to sign name mapping
- `asserts/*.png` - Sign icon images

---

## 🔧 Module 4: Pothole Detection

### Architecture
Pothole detection uses YOLO (You Only Look Once) object detection model to identify road surface defects in real-time.

**Processing Pipeline:**
```
Video Frame → YOLO Inference → Non-Max Suppression → 
Confidence Filtering → Bounding Box Annotation → Warning Overlay
```

### Technical Details

#### Model
- **Framework**: Ultralytics YOLO
- **Model File**: `best.pt` (PyTorch format)
- **Input**: Full-resolution color frames (BGR format)
- **Confidence Threshold**: 0.10 (10% minimum confidence)
- **Detection Trigger**: Only displays when > 2 potholes detected simultaneously

#### Detection Process
1. **YOLO Inference**: Single forward pass through YOLO network
2. **Post-processing**: Supervision library handles NMS and filtering
3. **Annotation**: Blue bounding boxes with labels
4. **Warning Display**: Red "POTHOLE DETECTED!" text overlay

#### Integration
- Integrated within `LaneLines` class for simultaneous processing
- Processes original color frame (before lane detection transformations)
- Runs independently but shares frame with lane detection pipeline
- **Feature**: Vision-based pothole detector using image processing + learned model
- **Proactive Maintenance**: Helps in proactive maintenance and driver awareness
- **Robust Localization**: Works in tandem with lane detection for robust localization

### Visual Display
- **Bounding Boxes**: Blue (#0055FF), 2px thickness
- **Labels**: Format: `class_name: confidence`
- **Warning Text**: Red text at top-center of frame
- **Font**: Hershey Simplex, size 1.0, thickness 3

### Libraries Used
- `ultralytics` - YOLO model inference
- `supervision` - Detection post-processing and annotation
- `opencv-python` - Image operations
- `numpy` - Array handling

### Key Files
- `LaneLines.py` - Contains pothole detection integration
- `pothole_main.py` - Standalone pothole detection demo
- `best.pt` - YOLO model weights

---

## 🔧 Module 5: Pedestrian Detection

### Architecture
Pedestrian detection uses Histogram of Oriented Gradients (HOG) with a linear SVM classifier for real-time human detection.

**Processing Pipeline:**
```
Video Frame → HOG Feature Extraction → Sliding Window Detection → 
Multi-scale Detection → Non-Max Suppression → Bounding Box Annotation
```

### Technical Details

#### HOG Descriptor
- **Default People Detector**: OpenCV's pre-trained HOG+SVM model
- **Feature Extraction**: 3780-dimensional feature vector per detection window
- **Window Stride**: (8, 8) pixels (optimized for speed)
- **Padding**: (4, 4) pixels
- **Scale Factor**: 1.05 (multi-scale detection)

#### Detection Parameters
- **Win Stride**: Controls sliding window step size (larger = faster, less accurate)
- **Scale**: Detector scales (1.05 = 5% increments)
- **Overlap Threshold**: 0.65 for Non-Max Suppression
- **Color**: Purple boxes (139, 34, 104) for visibility

#### Post-Processing
- **Non-Max Suppression**: Removes overlapping detections
- **Format Conversion**: Converts (x, y, w, h) to (x1, y1, x2, y2)
- **Confidence Matching**: Preserves detection confidence scores

### Visual Display
- **Bounding Boxes**: Purple, 2px thickness
- **Labels**: Format: `P{count}: {confidence}`
- **Count Display**: Total person count at bottom-left
- **Background**: Colored rectangle behind label text

### Key Features
- **Feature**: Real-time human detection module using lightweight detector
- **Prioritization**: Prioritizes close-range pedestrians and movement cues
- **Visual Warnings**: Generates visual warnings and can trigger audio alerts
- **Low False-Positive Rate**: Optimized for low false-positive rate in traffic scenarios

### Libraries Used
- `opencv-python` - HOG descriptor and people detector
- `imutils` - Non-max suppression utility
- `numpy` - Array operations

### Key Files
- `Human_Detection.py` - `HumanDetector` class implementation
- Integrated in `sign_lane.py` via `FindLaneLines` class

---

## ⚠️ Limitations

While the ADAS system provides comprehensive driver assistance, there are some limitations to be aware of:

- **Camera Quality Dependency**: Performance depends on camera quality and lighting conditions
- **Occlusion Challenges**: False positives possible under occlusion or extreme viewing angles
- **Audio Latency**: Audio playback latency depends on OS/driver (preloading reduces but doesn't eliminate it)
- **Device Performance**: Further model pruning may be needed for low-power devices (Raspberry Pi, mobile)
- **Environmental Factors**: Performance may degrade in extreme weather conditions (heavy rain, fog, snow)
- **Model Accuracy**: Detection accuracy varies with training data quality and scene complexity

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- FFmpeg (for audio playback in drowsiness detection)
- Webcam or video file for input

### System Dependencies

#### macOS
```bash
brew install portaudio  # For audio support
brew install ffmpeg     # For MP3 audio decoding
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install python3-opencv portaudio19-dev ffmpeg
```

#### Windows
- Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add FFmpeg to system PATH

### Python Dependencies

1. **Clone the repository:**
```bash
git clone https://github.com/MouryaVardhan04/Advance-Driving-Assistant-System.git
cd Advance-Driving-Assistant-System
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

### Required Python Packages
```
opencv-python
numpy
tensorflow
mediapipe
scipy>=1.14.0
pygame==2.5.0
pandas
pydub
simpleaudio
ultralytics
supervision
imutils
matplotlib
```

---

## 🎮 How to Run

### Option 1: Run All Modules Together (Recommended)

The main entry point runs both driver monitoring and road detection in parallel processes:

```bash
python main.py
```

This starts:
- **Process 1**: Drowsiness Detection (webcam input)
- **Process 2**: Lane + Sign + Pothole + Pedestrian Detection (video file)

### Option 2: Run Individual Modules

#### Drowsiness Detection Only
```bash
python combined_script.py
```
- Uses webcam (index 0)
- Press 'q' to quit
- Displays drowsiness level and triggers audio alarm

#### Lane + Sign + Pothole + Pedestrian Detection
```bash
python sign_lane.py video.mp4
```
- Replace `video.mp4` with your video file path
- Processes video with all road detection modules
- Press ESC or close window to quit

#### Pothole Detection Only (Demo)
```bash
python pothole_main.py
```
- Standalone pothole detection demo
- Uses `demo.mp4` by default
- Displays FPS and detection count

### Video File Configuration

Edit the following in `main.py` or `sign_lane.py`:
```python
# In main.py
p2 = Process(target=run_sign_lane, args=("video.mp4",), ...)

# Or in sign_lane.py, modify VIDEO_PATH variable
VIDEO_PATH = "your_video.mp4"
```

### Webcam Configuration

For drowsiness detection, the webcam index can be changed in `combined_script.py`:
```python
cap = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc. for different cameras
```

---

## 📁 Project Structure

```
ADAS/
├── main.py                      # Main entry point (multi-process)
├── combined_script.py           # Drowsiness detection module
├── sign_lane.py                 # Lane + Sign + Pothole + Pedestrian detection
├── LaneLines.py                 # Core lane detection class
├── Human_Detection.py           # Pedestrian detection class
├── pothole_main.py              # Standalone pothole detection demo
├── CameraCalibration.py         # Camera calibration utilities
├── Thresholding.py              # Color space thresholding
├── PerspectiveTransformation.py # Perspective transform utilities
├── fps_control.py               # Frame rate control utilities
│
├── Models/
│   └── sign_model.h5           # Pre-trained road sign CNN model
├── camera_cal/                 # Chessboard calibration images
├── asserts/                    # Road sign icon images (0.png - 42.png)
├── test_images/                # Test images for validation
│
├── best.pt                     # YOLO pothole detection model
├── beep.mp3                    # Drowsiness alarm audio file
├── labels.csv                  # Road sign class labels
├── requirements.txt            # Python dependencies
│
└── README.md                   # This file
```

---

## ⚙️ Configuration

### Drowsiness Detection Parameters

Edit constants in `combined_script.py`:
```python
EAR_THRESHOLD = 0.25              # Eye closure threshold
MAR_THRESHOLD = 0.6               # Yawn detection threshold
EAR_DURATION_ALERT_SEC = 1.0      # Seconds before alarm triggers
MAR_DURATION_ALERT_SEC = 2.0      # Seconds before yawn alert
RECOVERY_TIME_SEC = 20.0          # Seconds for recovery countdown
ALARM_SOUND_FILE = 'beep.mp3'     # Audio alarm file path
```

### Lane Detection Parameters

Edit in `LaneLines.py`:
```python
self.min_lane_pixels = 1200       # Minimum pixels for valid lane
self.min_lane_width = 200         # Minimum lane width (pixels)
self.max_lane_width = 1000        # Maximum lane width (pixels)
self.nwindows = 9                 # Number of sliding windows
self.margin = 80                  # Window margin (pixels)
```

### Pothole Detection Parameters

Edit in `sign_lane.py`:
```python
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.10       # YOLO confidence threshold
```

### Pedestrian Detection Parameters

Edit in `Human_Detection.py`:
```python
self.winStride = (8, 8)           # Sliding window stride
self.scale = 1.05                 # Multi-scale factor
self.overlapThresh = 0.65         # NMS overlap threshold
```

---

## 🔍 Performance Optimization

### For Real-time Performance

1. **Reduce Video Resolution**: Resize input frames before processing
2. **Lower Frame Rate**: Process every Nth frame
3. **Disable Unused Modules**: Comment out modules not needed
4. **GPU Acceleration**: Use CUDA-enabled PyTorch for YOLO (if available)

### Frame Rate Control

The project includes `fps_control.py` for maintaining consistent playback speed:
- Syncs to source video FPS
- Allows frame dropping if processing is slow
- Prevents video playback speedup

---

## 🐛 Troubleshooting

### Audio Alarm Not Working
- **Issue**: No sound plays during drowsiness detection
- **Solution**: 
  1. Install FFmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
  2. Verify `beep.mp3` exists in project root
  3. Check audio device permissions

### Camera Not Opening
- **Issue**: "Could not open video stream"
- **Solution**:
  1. Check camera index (try 0, 1, 2...)
  2. Ensure no other application is using the camera
  3. On Linux: `sudo usermod -a -G video $USER` (logout/login)

### Model Files Not Found
- **Issue**: "Could not load model"
- **Solution**:
  1. Ensure `best.pt` exists for pothole detection
  2. Ensure `Models/sign_model.h5` exists for sign detection
  3. Check file paths in code match actual locations

### Lane Detection Not Working
- **Issue**: Lanes not detected or incorrect
- **Solution**:
  1. Recalibrate camera with chessboard images
  2. Adjust threshold values in `Thresholding.py`
  3. Modify perspective transform points for your camera setup
  4. Ensure good lighting and clear lane markings

### Low FPS
- **Issue**: Slow performance
- **Solution**:
  1. Reduce video resolution
  2. Process fewer frames per second
  3. Disable heavy modules temporarily
  4. Use GPU acceleration if available

---

## 📊 System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4 GB
- **Camera**: USB webcam (720p recommended)
- **OS**: Windows 10, macOS 10.14+, or Ubuntu 18.04+

### Recommended Requirements
- **CPU**: Quad-core 3.0 GHz or higher
- **RAM**: 8 GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for YOLO acceleration)
- **Camera**: 1080p webcam for better accuracy

---

## 🧪 Testing

### Test Individual Modules

1. **Test Lane Detection:**
```bash
python sign_lane.py test_images/test1.jpg
```

2. **Test Drowsiness Detection:**
   - Run `combined_script.py` and close/open eyes
   - Verify alarm triggers after 1 second

3. **Test Sign Detection:**
   - Use video with clear road signs
   - Verify signs display when confidence ≥ 75%

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement
- GPU acceleration for YOLO inference
- Additional road sign classes
- Night vision support
- Mobile deployment (Android/iOS)
- Cloud logging and analytics

---

## 📧 Contact & Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

## 🙏 Acknowledgments

- MediaPipe team for Face Mesh solution
- Ultralytics for YOLO implementation
- OpenCV community for computer vision tools
- TensorFlow/Keras for deep learning framework

---

## 🎯 Conclusion & Next Steps

### Current Status
This integrated ADAS prototype provides immediate, practical safety benefits through its multi-module architecture. The drowsiness alarm, sign recognition, pothole detection, pedestrian detection, and lane modules work together seamlessly to create a comprehensive driver assistance system.

### Future Enhancements

#### Short-term Goals
- **Mobile/Raspberry Pi Deployments**: Optimize for embedded systems and mobile platforms
- **Model Compression**: Reduce model size for faster inference on low-power devices
- **Field Tests**: Conduct real-world testing in various driving conditions
- **Performance Optimization**: Further optimize frame processing for real-time performance

#### Long-term Vision
- **Vehicle CAN Integration**: Integrate with vehicle CAN bus for enhanced data fusion
- **Cloud Logging & Analytics**: Add cloud-based logging for fleet management and analytics
- **Additional Features**: 
  - Night vision support
  - Weather condition adaptation
  - Advanced driver behavior analysis
  - Integration with navigation systems
- **Feature Tuning**: Continuous improvement based on user feedback and field data

### Open for Contribution
The project is open for feature tuning, integration improvements, and collaboration. We welcome contributions that enhance safety, performance, and usability.

---

**Last Updated**: 2025
**Version**: 1.0
