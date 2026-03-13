import cv2
import numpy as np
import random
from old_tracker import TrackingData

class DotGridEffect:
    def __init__(self):
        # Grid settings
        self.step = 12                 # Space between dots
        self.max_displacement = 25     # How far the dots "pop out" in 3D
        self.bg_dot_color = (40, 40, 40) # Dim gray for the inactive background grid

    def _get_random_color(self):
        # Generates vibrant, random, changing colors (avoiding dark muddy colors)
        return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def _draw_landmark_dots(self, frame, landmarks, gray, h, w, max_offset, dot_size):
        if landmarks is None: return
        
        for lm in landmarks:
            lx, ly = int(lm[0] * w), int(lm[1] * h)
            
            # Clamp coordinates to prevent sampling errors
            safe_x = max(0, min(w - 1, lx))
            safe_y = max(0, min(h - 1, ly))
            
            # Get pixel brightness to calculate 3D "Z" depth
            intensity = gray[safe_y, safe_x]
            
            # Displace the dot up and left to create the 3D pop effect
            z_offset = int((intensity / 255.0) * max_offset)
            nx = safe_x - int(z_offset * 0.5)
            ny = safe_y - z_offset
            
            cv2.circle(frame, (nx, ny), dot_size, self._get_random_color(), -1)

    def process_frame(self, frame: np.ndarray, tracking_data: TrackingData = None) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # We start with a completely black canvas
        output = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get grayscale image to use as a 3D depth map
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Prepare the mask
        mask = None
        if tracking_data is not None and tracking_data.has_person:
            if tracking_data.segmentation_mask is not None:
                mask = tracking_data.segmentation_mask
                if len(mask.shape) > 2: mask = mask.squeeze()
                if mask.shape != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- A. Draw the Base Grid & 3D Body Displacement ---
        for y in range(0, h, self.step):
            for x in range(0, w, self.step):
                
                is_person = False
                if mask is not None:
                    # Check if the current grid coordinate is inside the human silhouette
                    is_person = mask[y, x] > 128

                if is_person:
                    # Sample brightness for depth
                    intensity = gray[y, x]
                    
                    # Calculate 3D extrusion (brighter pixels pop out further)
                    displacement = int((intensity / 255.0) * self.max_displacement)
                    
                    # Shift the coordinate to create an isometric 3D illusion
                    nx = x - int(displacement * 0.5)
                    ny = y - displacement
                    
                    # Brighter areas get slightly larger dots for volume
                    dot_radius = int(2 + (intensity / 255.0) * 3)
                    
                    cv2.circle(output, (nx, ny), dot_radius, self._get_random_color(), -1)
                else:
                    # Draw the flat, undisturbed 2D background grid
                    cv2.circle(output, (x, y), 1, self.bg_dot_color, -1)

        # --- B. Draw High-Density Face & Hands ---
        # If we only used the 12px grid above, facial features would be lost.
        # By iterating directly over the MediaPipe landmarks, we create a dense, 
        # highly-detailed 3D dot mesh specifically for the face and fingers.
        if tracking_data is not None and tracking_data.has_person:
            
            # Face (468 points) - Smaller dots for high detail
            self._draw_landmark_dots(output, tracking_data.face_landmarks, gray, h, w, 
                                     max_offset=self.max_displacement, dot_size=2)
            
            # Hands (21 points each) - Slightly larger dots
            self._draw_landmark_dots(output, tracking_data.left_hand_landmarks, gray, h, w, 
                                     max_offset=self.max_displacement, dot_size=3)
            self._draw_landmark_dots(output, tracking_data.right_hand_landmarks, gray, h, w, 
                                     max_offset=self.max_displacement, dot_size=3)

        return output

    def close(self):
        pass

if __name__ == "__main__":
    print("Run this via main.py to pass the MediaPipe tracking_data into the processor.")