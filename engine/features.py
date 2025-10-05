import pygame
import os
import eel
import re
import pywhatkit as kit
from engine.config import ASSISTANT_NAME  # keep if used elsewhere

@eel.expose
def playAssistantSound():
    """Plays the introductory sound for the assistant."""
    try:
        pygame.mixer.init()
        audio_path = os.path.join(os.path.dirname(__file__), "..", "www", "assets", "audio", "start_sound.mp3")
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing assistant sound: {e}")
        try:
            os.system('afplay /System/Library/Sounds/Glass.aiff')
        except:
            pass

def extract_yt_term(command):
    """Extract search term for YouTube."""
    pattern = r"play\s+(.*?)\s+on\s+youtube"
    match = re.search(pattern, command, re.IGNORECASE)
    return match.group(1).strip() if match else None

def PlayYoutube(query, speak_func):
    """Plays a video on YouTube based on the query."""
    search_term = extract_yt_term(query)
    if search_term:
        message = f"Playing {search_term} on YouTube"
        print(message)
        eel.DisplayMessage(message)
        speak_func(message)
        kit.playonyt(search_term)
    else:
        error_msg = "Sorry, I couldn't understand what to play."
        print(error_msg)
        eel.DisplayMessage(error_msg)
        speak_func(error_msg)
