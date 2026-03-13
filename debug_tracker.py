import sys
import os
import cv2
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from old_tracker import PoseTracker
from camera import CameraStream

def main():
    camera = CameraStream(source=0)
    camera.start()
    
    # Wait for camera to warm up
    time.sleep(2.0)
    
    tracker = PoseTracker()
    
    found = False
    for i in range(100):
        frame = camera.get_latest_frame()
        if frame is not None:
            data, vis = tracker.process_frame(frame, return_visualization=True)
            if data.has_person:
                print("--- PERSON FOUND ---")
                print("Landmarks len:", len(data.landmarks))
                print("Visibilities:", data.visibility[:5])
                found = True
                break
        time.sleep(0.05)
        
    if not found:
        print("--- NO PERSON DETECTED ---")
        
    camera.stop()

if __name__ == "__main__":
    main()
