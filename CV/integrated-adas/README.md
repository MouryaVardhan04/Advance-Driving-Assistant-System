# integrated-adas/README.md

# Integrated ADAS (Advanced Driver Assistance System)

This project integrates lane detection, road sign detection, drowsiness detection, and emotion recognition to enhance driver safety and experience.

## Project Structure

```
integrated-adas
├── src
│   ├── detection
│   │   ├── lane_detection.py       # Lane detection algorithms
│   │   ├── sign_detection.py       # Road sign detection algorithms
│   │   ├── drowsiness_detection.py  # Drowsiness detection algorithms
│   │   └── emotion_detection.py     # Emotion recognition algorithms
│   ├── models
│   │   ├── sign_model.h5           # Pre-trained model for sign detection
│   │   ├── drowsiness_model.h5      # Pre-trained model for drowsiness detection
│   │   └── emotion_model.h5        # Pre-trained model for emotion recognition
│   ├── utils
│   │   ├── camera_calibration.py    # Camera calibration functions
│   │   ├── image_processing.py       # Image processing utilities
│   │   └── visualization.py          # Visualization functions
│   ├── config
│   │   ├── settings.py              # Configuration settings
│   │   └── labels.csv               # Labels for detected signs and emotions
│   └── main.py                      # Main application logic
├── tests
│   └── test_detections.py           # Unit tests for detection modules
├── requirements.txt                  # Project dependencies
├── setup.py                          # Packaging information
└── README.md                         # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd integrated-adas
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/main.py
   ```

## Usage

- The application processes video input to detect lanes, road signs, drowsiness, and emotions.
- Ensure that the necessary models are available in the `src/models` directory.

## Modules

- **Lane Detection**: Identifies lane markings in images.
- **Sign Detection**: Recognizes traffic signs from images.
- **Drowsiness Detection**: Evaluates driver alertness based on facial features.
- **Emotion Recognition**: Classifies emotions from facial expressions.

## Testing

Run the unit tests to ensure the detection algorithms work as expected:
```
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License.