# filepath: /integrated-adas/integrated-adas/src/config/settings.py
MODEL_PATHS = {
    "sign": "src/models/sign_model.h5",
    "drowsiness": "src/models/drowsiness_model.h5",
    "emotion": "src/models/emotion_model.h5"
}

DETECTION_THRESHOLDS = {
    "lane": 0.5,
    "sign": 0.7,
    "drowsiness": 0.6,
    "emotion": 0.5
}

CAMERA_SETTINGS = {
    "width": 640,
    "height": 480,
    "fps": 30
}