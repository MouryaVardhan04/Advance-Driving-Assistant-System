# main.py
from multiprocessing import Process
from CV.drowsiness import start_application
from road_main import run_detector as run_roadsign_detector

if __name__ == '__main__':
    # Create processes
    p1 = Process(target=start_application)
    p2 = Process(target=run_roadsign_detector)

    # Start both
    p1.start()
    p2.start()

    # Wait for both to finish
    p1.join()
    p2.join()
