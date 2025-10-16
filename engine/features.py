import pygame
import os
import eel
import re 
# NOTE: Assuming ASSISTANT_NAME is defined in engine.config, using placeholder for completeness
# from engine.config import ASSISTANT_NAME 
ASSISTANT_NAME = "jarvis" # Placeholder in case config is missing

# 🌟 CRITICAL FIX: Use relative import to fix the ImportError
from .command import speak 
import pywhatkit as kit

@eel.expose
def playAssistantSound():
    try:
        pygame.mixer.init()
        audio_path = os.path.join(os.path.dirname(__file__), "..", "www", "assets", "audio", "start_sound.mp3")
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    except Exception as e:
        print(f"Error playing assistant sound: {e}")
        try:
            os.system('afplay /System/Library/Sounds/Glass.aiff')
        except:
            pass

def openCommand(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("open", "")
    query = query.strip().lower()

    APP_ALIASES = {
        "brave": "Brave Browser",
        "chrome": "Google Chrome",
        "google": "Google Chrome",
        "safari": "Safari",
        "vs code": "Visual Studio Code",
        "spotify": "Spotify",
        "pages": "Pages",
        "whatsup": "WhatsApp",
    }

    app_name = APP_ALIASES.get(query, query)
    print(f"User query: \"{query}\"")
    eel.DisplayMessage(f"Mapped app name: \"{app_name}\"")

    try:
        message = f"Opening {app_name}"
        eel.DisplayMessage(message)
        speak(message)
        exit_code = os.system(f'open -a "{app_name}"')
        print(f"Open command exit code: {exit_code}")
        if exit_code != 0:
            raise Exception(f"Failed to open {app_name}, exit code {exit_code}")
    except Exception as e:
        print(f"❌ Error in openCommand: {e}")
        speak(f"Sorry, I couldn't open {query}")

def PlayYoutube(query):
    search_term = extract_yt_term(query)
    if search_term:
        message = f"Playing {search_term} on YouTube"
        print(message)
        eel.DisplayMessage(message)
        speak(message)
        kit.playonyt(search_term)
    else:
        error_msg = "Sorry, I couldn't understand what to play."
        print(error_msg)
        eel.DisplayMessage(error_msg)
        speak(error_msg)

def extract_yt_term(command):
    pattern = r"play\s+(.*?)\s+on\s+youtube"
    match = re.search(pattern, command, re.IGNORECASE)
    return match.group(1) if match else None
