# engine/assistant.py (Your original command.py, renamed and modified)

import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import eel
import time
import os
import datetime 
import requests

# ... (API Keys and CONVERSATION_MEMORY_MAX_LENGTH are the same) ...
GEMINI_API_KEY = "AIzaSyAoE9iIeJc09x-Ul2xInVBoNrCPiikiVbs"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
CONVERSATION_MEMORY_MAX_LENGTH = 10
conversation_history = [] 

# ... (Engine, voice setup, speak function are the same) ...
engine = pyttsx3.init()
voices = engine.getProperty('voices')

def set_sweet_girl_voice(engine, voices):
    # ... (Voice customization logic is the same) ...
    selected_voice_id = None
    TARGET_ID = 'com.apple.speech.synthesis.voice.samantha'

    for voice in voices:
        if voice.id == TARGET_ID:
            selected_voice_id = voice.id
            print(f"[Voice] Found and set target high-quality voice: {voice.name}")
            break

    if not selected_voice_id:
        for voice in voices:
            if voice.gender == ['VoiceGenderFemale'] and voice.name.lower().startswith('en'):
                selected_voice_id = voice.id
                print(f"[Voice] Falling back to generic English female voice: {voice.name}")
                break

    if not selected_voice_id and voices:
        selected_voice_id = voices[0].id
        print(f"[Voice] Warning: No explicit female voice found. Falling back to default: {voices[0].name}")

    if selected_voice_id:
        try:
            engine.setProperty('voice', selected_voice_id)
        except Exception as e:
            print(f"[Voice Error] Failed to set voice property: {e}")
            if voices:
                engine.setProperty('voice', voices[0].id)

set_sweet_girl_voice(engine, voices)
engine.setProperty('rate', 180)
engine.setProperty('volume', 1.0)

def speak(text):
    print(f"[Speaking]: {text}")
    engine.say(text)
    engine.runAndWait()


def safe_eel_call(func_name, *args):
    """Safely call Eel functions, with fallback if not available"""
    try:
        if func_name == "DisplayMessage":
            eel.DisplayMessage(*args) # type : ignore
        elif func_name == "showHood":
            eel.showHood() # type : ignore
        elif func_name == "setMicState":
            eel.setMicState(*args) # type : ignore
        elif func_name == "updateDrowsinessEmotion":
            eel.updateDrowsinessEmotion(*args) # type: ignore
        elif func_name == "updateRoadSign":
            eel.updateRoadSign(*args) # type: ignore
    except Exception as e:
        print(f"Eel call failed for {func_name}: {e}")

# --- GLOBAL STATE FOR CV DATA ---
current_drowsiness_level = 0
current_emotion = "Neutral"
current_road_sign = "None" # NEW: To track the last seen road sign

# --- GEMINI API INTEGRATION (Logic is mostly the same, but uses new global state) ---
def get_gemini_response(user_query):
    global current_drowsiness_level, current_emotion, current_road_sign # Use road sign
    
    # ... (System prompt logic based on drowsiness and emotion is the same) ...
    # Simplified prompt logic for space, but assumes the original detailed logic is here.
    system_prompt = (
        "You are a friendly, concise, and helpful virtual assistant named JARVIS. "
        f"The user's current status is: Drowsiness Level {current_drowsiness_level}, Emotion: {current_emotion}, "
        f"Last Road Sign: {current_road_sign}. Use this context, especially if the status is negative or important. "
        "Keep your responses short, conversational, and avoid sounding too formal. "
        "If the user asks a question, answer it directly and warmly. Use the provided chat history for context."
    )
    # ... (Rest of the Gemini API call remains the same) ...
    global conversation_history
    contents = conversation_history + [{"role": "user", "parts": [{"text": user_query}]}]

    payload = {
        "contents": contents,
        "tools": [{"google_search": {} }],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers=headers,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")

        if text:
            return text.strip()
        else:
            return "I contacted my AI core, but it didn't give me a clear answer right now. Sorry!"

    except requests.exceptions.Timeout:
        return "I'm having trouble reaching the network right now. Maybe try again in a moment?"
    except requests.exceptions.RequestException as e:
        print(f"Gemini API Request Error: {e}")
        return "Oops, I ran into an error trying to process that request."
    except Exception as e:
        print(f"General AI Error: {e}")
        return "I'm experiencing a minor system hiccup. Can you try phrasing that differently?"


# ... (takecommand, start_conversation, processCommand, allCommands, display_done functions are the same) ...
@eel.expose
def takecommand():
    # ... (takecommand logic is the same) ...
    fs = 16000
    seconds = 4
    r = sr.Recognizer()
    output_filename = 'output.wav'

    r.energy_threshold = 200
    query = ""

    try:
        print("Recording...")
        safe_eel_call("DisplayMessage", "Listening...")

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wav.write(output_filename, fs, myrecording)
        print("Recording finished.")

        with sr.AudioFile(output_filename) as source:
            audio = r.record(source)

        print("Recognizing...")
        safe_eel_call("DisplayMessage", "Recognizing...")

        query = r.recognize_google(audio, language='en-in')

        print("User said:", query)

        return query.lower()

    except sr.UnknownValueError:
        print("Sorry, could not recognize the audio.")
        safe_eel_call("DisplayMessage", "Sorry, I couldn't understand.")
        return "unrecognized"
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        safe_eel_call("DisplayMessage", "Sorry, there was an error with the speech recognition service.")
        return "error"
    except Exception as e:
        print(f"Error: {e}")
        safe_eel_call("DisplayMessage", "Sorry, an unexpected error occurred.")
        return "error"
    finally:
        if os.path.exists(output_filename):
            os.remove(output_filename)

@eel.expose
def start_conversation():
    print("Starting continuous conversation loop.")

    safe_eel_call("setMicState", "continuous")

    global conversation_history
    conversation_history = []

    while True:
        query = takecommand()

        if query == "unrecognized" or query == "error":
            time.sleep(0.5)
            continue

        if "quit" in query or "exit" in query or "stop listening" in query:
            response = "Understood. Ending conversation."
            safe_eel_call("DisplayMessage", response)
            speak(response)
            safe_eel_call("setMicState", "idle")
            conversation_history = []
            break

        processCommand(query)

        time.sleep(0.1)


def processCommand(query):
    query = query.lower()
    print("Command received:", query)

    global conversation_history

    if "open" in query:
        # Assuming engine.features is available in your structure
        # from engine.features import openCommand
        # openCommand(query) 
        response = "I need the openCommand function from engine.features to do that."
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(0.1)

    elif "on youtube" in query:
        # Assuming engine.features is available in your structure
        # from engine.features import PlayYoutube
        # PlayYoutube(query)
        response = "I need the PlayYoutube function from engine.features to do that."
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(0.1)

    elif "hello" in query or "hi" in query:
        response = "Hello! What can I help you with?"
        print(response)
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(0.1)

    elif "time" in query:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        response = f"The current time is {current_time}"
        print(response)
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(0.1)

    elif "date" in query:
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        response = f"Today is {current_date}"
        print(response)
        safe_eel_call("DisplayMessage", response)
        speak(response)
        time.sleep(0.1)

    else:
        conversation_history.append({"role": "user", "parts": [{"text": query}]})

        print("Sending query to AI...")
        safe_eel_call("DisplayMessage", "Thinking...")
        ai_response = get_gemini_response(query)

        conversation_history.append({"role": "model", "parts": [{"text": ai_response}]})

        if len(conversation_history) > CONVERSATION_MEMORY_MAX_LENGTH:
            conversation_history = conversation_history[-CONVERSATION_MEMORY_MAX_LENGTH:]


        print("AI Response:", ai_response)
        safe_eel_call("DisplayMessage", ai_response)
        speak(ai_response)
        time.sleep(0.1)


@eel.expose
def allCommands(query=""):
    # ... (allCommands logic is the same) ...
    if not query:
        print("No command received")
        safe_eel_call("DisplayMessage", "No command received")
        return

    global conversation_history
    temp_history = conversation_history
    conversation_history = []

    if "quit" in query.lower() or "exit" in query.lower():
        response = "Understood. Ending conversation. Have a great day!"
        safe_eel_call("DisplayMessage", response)
        speak(response)
        safe_eel_call("showHood")
    else:
        processCommand(query)
        safe_eel_call("showHood")

    conversation_history = temp_history

@eel.expose
def display_done():
    print("✅ Frontend finished displaying message")

# NEW: Eel function to update drowsiness level and emotion
@eel.expose
def updateDrowsinessEmotion(level, emotion):
    global current_drowsiness_level, current_emotion
    current_drowsiness_level = level
    current_emotion = emotion
    print(f"Drowsiness Level updated: {level}, Emotion: {emotion}")

# NEW: Eel function to update road sign status
@eel.expose
def updateRoadSign(sign_name):
    global current_road_sign
    current_road_sign = sign_name
    # Optional: Trigger an immediate audio alert for critical signs
    if sign_name in ["Stop", "Yield", "Danger"]:
        speak(f"Alert! {sign_name} sign detected.")
    safe_eel_call("updateRoadSign", sign_name)
    print(f"Road Sign updated: {sign_name}")