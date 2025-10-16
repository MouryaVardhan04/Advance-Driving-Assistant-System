import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import eel
import time
import os
import datetime 
import requests # Required for the Gemini API call

# NOTE: Using the API key provided by the user for demonstration.
GEMINI_API_KEY = "AIzaSyAoE9iIeJc09x-Ul2xInVBoNrCPiikiVbs"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# 🌟 NEW GLOBAL STATE: Stores conversation history (User and Model turns)
# Max memory length is 10 turns (5 user inputs + 5 model outputs) to save tokens.
CONVERSATION_MEMORY_MAX_LENGTH = 10
conversation_history = [] 

# ✅ Initialize engine globally
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# --- VOICE CUSTOMIZATION START ---
def set_sweet_girl_voice(engine, voices):
    selected_voice_id = None
    
    # --- 🎯 TARGET MAC HIGH-QUALITY FEMALE VOICE ID ---
    # 1. Target "Samantha" (a standard high-quality Mac voice)
    TARGET_ID = 'com.apple.speech.synthesis.voice.samantha'
    
    # Alternative choices for Mac:
    # 'com.apple.speech.synthesis.voice.ava' (More modern)
    # 'com.apple.speech.synthesis.voice.zoe' (Child-like/high pitch)

    # 1. Try to find the specific high-quality Mac voice
    for voice in voices:
        if voice.id == TARGET_ID:
            selected_voice_id = voice.id
            print(f"[Voice] Found and set target high-quality voice: {voice.name}")
            break

    # 2. Fallback: Search for any voice explicitly marked as Female
    if not selected_voice_id:
        for voice in voices:
            if voice.gender == ['VoiceGenderFemale'] and voice.name.lower().startswith('en'):
                selected_voice_id = voice.id
                print(f"[Voice] Falling back to generic English female voice: {voice.name}")
                break
    
    # 3. Final Fallback (if no female voice was found)
    if not selected_voice_id and voices:
        selected_voice_id = voices[0].id
        print(f"[Voice] Warning: No explicit female voice found. Falling back to default: {voices[0].name}")

    # 4. Apply the voice ID
    if selected_voice_id:
        try:
            engine.setProperty('voice', selected_voice_id)
        except Exception as e:
            print(f"[Voice Error] Failed to set voice property: {e}")
            if voices:
                engine.setProperty('voice', voices[0].id) 

set_sweet_girl_voice(engine, voices)
# --- VOICE CUSTOMIZATION END ---

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
    except Exception as e:
        print(f"Eel call failed for {func_name}: {e}")

# --- GEMINI API INTEGRATION ---
def get_gemini_response(user_query):
    """Fetches a friendly, conversational response from the Gemini API."""
    
    system_prompt = (
        "You are a friendly, concise, and helpful virtual assistant named JARVIS. "
        "Keep your responses short, conversational, and avoid sounding too formal. "
        "If the user asks a question, answer it directly and warmly. Use the provided chat history for context."
    )
    
    # 🌟 MEMORY INTEGRATION: Prepend conversation history to the current query
    global conversation_history
    contents = conversation_history + [{"role": "user", "parts": [{"text": user_query}]}]

    payload = {
        "contents": contents, # Pass the full history
        "tools": [{"google_search": {} }],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
            json=payload, 
            headers=headers,
            # CRITICAL FOR SPEED: Use a short timeout
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
# -----------------------------

@eel.expose
def takecommand():
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
        
        # We process the command immediately after recognizing
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
    """Starts the continuous listening loop."""
    print("Starting continuous conversation loop.")
    
    safe_eel_call("setMicState", "continuous") 
    
    # 🌟 MEMORY FIX: Clear memory when a new continuous conversation starts
    global conversation_history
    conversation_history = []
    
    while True:
        # 1. Take Command (Recognition)
        query = takecommand()
        
        if query == "unrecognized" or query == "error":
            time.sleep(0.5) # Short pause before the next loop
            continue 

        # 2. Check for termination commands
        if "quit" in query or "exit" in query or "stop listening" in query:
            response = "Understood. Ending conversation."
            safe_eel_call("DisplayMessage", response)
            speak(response)
            safe_eel_call("setMicState", "idle") 
            conversation_history = [] # Final clear upon exit
            break

        # 3. Process the command (Handles responses)
        processCommand(query)
        
        # Short delay before the next recording begins
        time.sleep(0.1)


def processCommand(query):
    """Process commands and provide responses"""
    query = query.lower()
    print("Command received:", query)
    
    global conversation_history
    
    if "open" in query:
        from engine.features import openCommand
        openCommand(query)
        time.sleep(0.1) 

    elif "on youtube" in query:
        from engine.features import PlayYoutube
        PlayYoutube(query)
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
        # 🌟 MEMORY: 1. Append user query to history
        conversation_history.append({"role": "user", "parts": [{"text": query}]})
        
        # 🌟 AI CONVERSATION: Send unknown commands to Gemini
        print("Sending query to AI...")
        safe_eel_call("DisplayMessage", "Thinking...")
        ai_response = get_gemini_response(query)
        
        # 🌟 MEMORY: 2. Append model response to history
        conversation_history.append({"role": "model", "parts": [{"text": ai_response}]})

        # 🌟 MEMORY: 3. Trim history to prevent it from getting too long
        if len(conversation_history) > CONVERSATION_MEMORY_MAX_LENGTH:
            # Keep only the last N turns (trimming the oldest 2 messages/1 turn)
            conversation_history = conversation_history[-CONVERSATION_MEMORY_MAX_LENGTH:]


        print("AI Response:", ai_response)
        safe_eel_call("DisplayMessage", ai_response)
        speak(ai_response)
        time.sleep(0.1)


@eel.expose
def allCommands(query=""):
    """Exposed function for frontend to handle text input directly."""
    if not query:
        print("No command received")
        safe_eel_call("DisplayMessage", "No command received")
        return
    
    # Text commands are typically stateless, so we temporarily clear the memory
    # so text input does not interfere with a waiting voice conversation's history.
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
    
    # Restore history (if there was an ongoing voice conversation)
    conversation_history = temp_history

@eel.expose
def display_done():
    """Called by frontend when display animation is complete"""
    print("✅ Frontend finished displaying message")
