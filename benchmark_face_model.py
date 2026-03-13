import os
import sys
import torch
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_engine import ArtGenerator
from old_tracker import TrackingData

def main():
    print("Testing CPU Face Stylization Performance...")
    # Initialize ArtGenerator with model_type="face" and device="cpu"
    generator = ArtGenerator(
        model_path=None, 
        input_size=(128, 128), 
        use_fp16=False, 
        device="cpu"
    )
    generator.model_type = "face" # explicitly set
    
    # Create dummy 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create dummy tracking data with face landmarks in the center
    # MediaPipe returns array of [x, y, z] normalized
    dummy_lms = np.array([
        [0.5, 0.5, 0.0], # Nose
        [0.45, 0.45, 0.0], # L Eye
        [0.55, 0.45, 0.0], # R Eye
        [0.4, 0.5, 0.0], # L Ear
        [0.6, 0.5, 0.0], # R Ear
    ])
    
    tracking_data = TrackingData()
    tracking_data.has_person = True
    tracking_data.landmarks = dummy_lms
    
    print("Warming up inference...")
    for _ in range(5):
        _ = generator.generate_with_compositing(frame, tracking_data)
        
    print("Benchmarking 10 frames...")
    start_time = time.time()
    for _ in range(10):
        _ = generator.generate_with_compositing(frame, tracking_data)
    
    total_time = time.time() - start_time
    avg_time_ms = (total_time / 10) * 1000
    fps = 1000.0 / avg_time_ms
    
    print(f"Average Face Compositing Inference: {avg_time_ms:.2f}ms")
    print(f"Theoretical FPS: {fps:.2f}fps")
    
if __name__ == "__main__":
    main()
