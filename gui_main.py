import eel

from engine.command import *
from engine.features import playAssistantSound 

eel.init("www") 

# Optional startup action
playAssistantSound()

eel.start("index.html", size=(1000, 700), block=True, mode='default')
