from multiprocessing import Process
# Assuming the combined script is saved as 'combined_script.py'
from combined_script import start_integrated_detection 
# Assuming road_main.py has a function named 'run_detector'
from road_main import run_detector as run_roadsign_detector 

if __name__ == '__main__':
    # Print a message to confirm startup of main process
    print("Starting Multi-Process Application...")
    
    # Process 1: Integrated Driver Monitoring (Drowsiness & Emotion)
    p1 = Process(target=start_integrated_detection, name="DriverMonitorProcess")
    
    # Process 2: Road Sign Detection
    p2 = Process(target=run_roadsign_detector, name="RoadSignProcess")

    # Start both processes
    p1.start()
    p2.start()
    print("All processes started successfully.")

    # Wait for both to finish (blocking)
    p1.join()
    p2.join()
    
    print("All monitoring processes have shut down.")