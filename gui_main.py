import eel

from engine.features import *  # If needed
from engine.command import *   # takecommand is exposed here

# Initialize Eel with the folder that contains index.html
eel.init("www")  # Make sure "www/index.html" exists

# Optional startup action
playAssistantSound()

# Start the Eel frontend and keep it running
eel.start("index.html", mode='safari', size=(1000, 700), block=True)
