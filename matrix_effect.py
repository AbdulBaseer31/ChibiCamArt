import cv2
import numpy as np
import random
from typing import Optional
from old_tracker import TrackingData

class MatrixEffect:
    def __init__(self):
        # We removed the hardcoded width/height. 
        # The effect will now dynamically adapt to whatever resolution main.py feeds it.
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.char_width = 10
        self.char_height = 14
        
        # State variables initialized to None; they will set up on the first frame
        self.canvas = None
        self.drops = []
        self.speeds = []
        self.columns = 0
    
    def _get_random_char(self):
        return random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    def process_frame(self, frame: np.ndarray, tracking_data: TrackingData = None) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # 1. Initialize or resize canvas dynamically to match the exact camera feed
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.columns = int(w / self.char_width)
            self.drops = [random.randint(0, h) for _ in range(self.columns)]
            self.speeds = [random.randint(1, 4) for _ in range(self.columns)]
        
        # 2. Fade the background canvas for the trail effect
        self.canvas = cv2.addWeighted(self.canvas, 0.85, np.zeros_like(self.canvas), 0.15, 0)

        # 3. Draw falling background rain
        for i in range(len(self.drops)):
            char = self._get_random_char()
            x = i * self.char_width
            y = self.drops[i] * self.char_height
            cv2.putText(self.canvas, char, (x, y), self.font, 0.4, (0, 200, 0), 1)
            
            if y > h and random.random() > 0.95:
                self.drops[i] = 0
            self.drops[i] += self.speeds[i]

        matrix_frame = self.canvas.copy()
        
        if tracking_data is not None and tracking_data.has_person:
            # Extract grayscale to map real-world facial shadows/highlights to text brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # -- A. Body (Using Segmentation Mask) --
            if tracking_data.segmentation_mask is not None:
                mask = tracking_data.segmentation_mask
                # Ensure mask perfectly matches current frame dimensions
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Draw a matrix grid over the body silhouette
                step = 12 
                for gy in range(0, h, step):
                    for gx in range(0, w, step):
                        if mask[gy, gx] > 128:
                            intensity = int(gray[gy, gx])
                            # Darken the body slightly so the face pops out more
                            color = (0, max(30, intensity - 20), 0)
                            char = self._get_random_char()
                            cv2.putText(matrix_frame, char, (gx, gy), self.font, 0.35, color, 1)

            # Helper function to plot exact landmarks with luminance sampling
            def plot_landmarks(landmarks, base_scale, color_boost=0, is_face=False):
                if landmarks is not None:
                    for lm in landmarks:
                        # Translate normalized [0,1] coordinates directly to native frame pixels
                        lx, ly = int(lm[0] * w), int(lm[1] * h)
                        if 0 <= lx < w and 0 <= ly < h:
                            intensity = int(gray[ly, lx])
                            
                            if is_face and intensity > 150:
                                # Highlight features like eyes/nose in a brighter whitish-green
                                color = (min(255, intensity), 255, min(255, intensity))
                            else:
                                color = (0, min(255, intensity + color_boost), 0)
                                
                            char = self._get_random_char()
                            cv2.putText(matrix_frame, char, (lx, ly), self.font, base_scale, color, 1)

            # -- B. Face (Reduced density) --
            # Slicing with [::3] skips points, drastically reducing how "busy" the face looks.
            # Change the 3 to a 2 for a slightly denser face, or 4 for an even sparser one.
            if tracking_data.face_landmarks is not None:
                plot_landmarks(tracking_data.face_landmarks[::3], base_scale=0.25, color_boost=40, is_face=True)

            # -- C. Hands (21 points each) --
            plot_landmarks(tracking_data.left_hand_landmarks, base_scale=0.4, color_boost=60)
            plot_landmarks(tracking_data.right_hand_landmarks, base_scale=0.4, color_boost=60)
            
        return matrix_frame
    
    def close(self):
        pass

if __name__ == "__main__":
    print("Run this via main.py to pass the MediaPipe tracking_data into the processor.")