from multiprocessing import Process
from combined_script import start_integrated_detection
from sign_lane import run_sign_lane

if __name__ == '__main__':
    print("Starting Multi-Process Application...")

    # Process 1: Integrated Driver Monitoring (Drowsiness & Emotion)
    p1 = Process(target=start_integrated_detection, name="DriverMonitorProcess")

    # Process 2: Combined Lane & Road Sign Detection
    p2 = Process(target=run_sign_lane, args=("project_video.mp4",), name="SignLaneProcess")

    # Start both processes
    p1.start()
    p2.start()
    print("All processes started successfully.")

    # Wait for both to finish (blocking)
    p1.join()
    p2.join()

    print("All monitoring processes have shut down.")