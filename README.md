# Jarvis Voice Assistant

A Python-based voice assistant with a web-based UI using Eel framework.

## Features

- Voice recognition using Google Speech Recognition API
- Text-to-speech output
- Web-based user interface
- Audio recording and playback
- Command processing system

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install system dependencies (macOS):**
   ```bash
   brew install portaudio
   ```

3. **For speech recognition to work, you need:**
   - Internet connection (for Google Speech Recognition API)
   - Microphone access

## Usage

1. **Run the application:**
   ```bash
   python main.py
   ```

2. **Using the voice assistant:**
   - Click the microphone button or press `Cmd+J` to start voice recognition
   - Speak your command clearly
   - The assistant will process your command and respond

3. **Text input:**
   - Type commands in the text box
   - Press Enter or click Send to submit

## Current Commands

- Commands containing "open" will trigger a response
- More commands can be added in `engine/command.py`

## Troubleshooting

- **Speech recognition not working:** Check your internet connection and microphone permissions
- **Audio issues:** Ensure your system audio is working and microphone is properly configured
- **Voice not working:** The system will automatically fall back to the first available voice

## Project Structure

```
Jarvis/
├── engine/
│   ├── command.py      # Voice recognition and command processing
│   └── features.py     # Audio playback and system features
├── www/                # Web UI files
│   ├── index.html
│   ├── main.js
│   ├── controller.js
│   └── assets/
├── main.py             # Application entry point
└── requirements.txt    # Python dependencies
``` 