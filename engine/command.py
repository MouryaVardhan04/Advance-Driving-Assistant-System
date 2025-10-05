import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import eel
import time
import os
import datetime
# Removed pyttsx3 and objc imports as they are no longer used for speech.
from engine.features import PlayYoutube, playAssistantSound

# ============================================================
# ✅ Initialize TTS (Bypassed pyttsx3, relying on native macOS 'say')
# ============================================================

# The pyttsx3 initialization logic has been entirely removed as it was buggy
# and the speak function no longer depends on it.

def speak(text: str):
    """Text-to-speech using native macOS 'say' command."""
    print(f"[Speaking]: {text}")
    try:
        # Uses native macOS 'say' command
        safe_text = text.replace("'", "").replace('"', '')
        os.system(f'say "{safe_text}"')
        # Removed time.sleep temporarily for testing, but typically useful.
    except Exception as e:
        print(f"TTS failed: {e}")

print("✅ TTS Initialized using native macOS 'say' command.")


# ============================================================
# ✅ Safe EEL Calls
# ============================================================
def safe_eel_call(func_name, *args):
    """Safely call Eel or Python functions."""
    try:
        if func_name == "DisplayMessage":
            eel.DisplayMessage(*args)
        elif func_name == "showHood":
            eel.showHood()
        elif func_name == "hideSiriWave":
            eel.hideSiriWave()
        elif func_name == "playAssistantSound":
            playAssistantSound()  # Direct Python call
    except NameError:
        print(f"[Eel Not Running] {func_name} call skipped.")
    except Exception as e:
        print(f"[Eel call failed] {func_name}: {e}")

# ============================================================
# ✅ Main Voice Command Function
# ============================================================
@eel.expose
def takecommand():
    fs = 16000
    seconds = 5
    print("🎙️ Recording...")
    safe_eel_call("DisplayMessage", "Listening...")
    safe_eel_call("playAssistantSound")
    try:
        # Records audio using sounddevice
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wav.write('output.wav', fs, myrecording)
        print("✅ Recording finished.")
    except Exception as e:
        print(f"Recording error: {e}")
        safe_eel_call("DisplayMessage", "Microphone not detected.")
        return ""

    r = sr.Recognizer()
    try:
        with sr.AudioFile('output.wav') as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        print("🧠 Recognizing...")
        safe_eel_call("DisplayMessage", "Recognizing...")
        
        # Recognize speech using Google's API
        query = r.recognize_google(audio, language='en-in')
        print("User said:", query)
        safe_eel_call("DisplayMessage", query)
        time.sleep(0.3)
        processCommand(query)
        return query.lower()
        
    except sr.UnknownValueError:
        safe_eel_call("DisplayMessage", "Sorry, I couldn't understand.")
        speak("Sorry, I couldn't understand.")
        safe_eel_call("showHood")
        return ""
    except sr.RequestError as e:
        safe_eel_call("DisplayMessage", "Speech recognition service error.")
        speak("Speech recognition service error.")
        safe_eel_call("showHood")
        return ""
    except Exception as e:
        safe_eel_call("DisplayMessage", "Unexpected error occurred.")
        speak("An unexpected error occurred.")
        safe_eel_call("showHood")
        return ""

# ============================================================
# ✅ Command Processing
# ============================================================
def processCommand(query):
    query = query.lower().strip()
    print("🧩 Command received:", query)

    if "on youtube" in query:
        try:
            PlayYoutube(query, speak)
        except Exception as e:
            print(f"YouTube error: {e}")
        time.sleep(1)
        safe_eel_call("showHood")

    elif "hello" in query or "hi" in query:
        response = "Hello! How can I help you?"
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(1)
        safe_eel_call("showHood")

    elif "time" in query:
        response = f"The time is {datetime.datetime.now().strftime('%I:%M %p')}"
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(1)
        safe_eel_call("showHood")

    elif "date" in query:
        response = f"Today is {datetime.datetime.now().strftime('%B %d, %Y')}"
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(1)
        safe_eel_call("showHood")

    else:
        response = "I don't understand that command. I can help with YouTube, time, or date."
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(1)
        safe_eel_call("showHood")

# ============================================================
# ✅ Eel-exposed helpers
# ============================================================
@eel.expose
def allCommands(query=""):
    if not query:
        safe_eel_call("DisplayMessage", "No command received.")
        return
    processCommand(query)

@eel.expose
def display_done():
    print("✅ Frontend finished displaying message.")

# ============================================================
# ✅ Main Execution
# ============================================================
if __name__ == "__main__":
    print("🎯 Voice assistant backend ready.")
    speak("Voice assistant initialized successfully.")
    eel.init('www')  # your web folder
    eel.start('index.html', size=(700, 500), port=8000)
