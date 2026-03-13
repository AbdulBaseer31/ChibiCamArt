import cv2
import numpy as np
import random
import math
from old_tracker import TrackingData

class DotGridEffect:
    def __init__(self):
        # --- Original Foreground Settings ---
        self.step = 5                  
        self.max_displacement = 30     
        self.time_offset = 0.0         

        # --- New Background Settings ---
        self.bg_step = 5               # Grid size same
        self.prev_gray = None          # Stores previous frame for movement detection
        self.flow_map = None           # Persistent buffer to hold the "flame" colors
        self.bg_dot_mask = None        # Pre-rendered dot mask to keep FPS high

    def _get_random_color(self):
        return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def _draw_landmark_dots(self, frame, landmarks, gray, h, w, max_offset, dot_size):
        if landmarks is None: return
        
        for lm in landmarks:
            lx, ly = int(lm[0] * w), int(lm[1] * h)
            safe_x = max(0, min(w - 1, lx))
            safe_y = max(0, min(h - 1, ly))
            
            intensity = gray[safe_y, safe_x]
            z_offset = int((intensity / 255.0) * max_offset)
            nx = safe_x - int(z_offset * 0.5)
            ny = safe_y - z_offset
            
            cv2.circle(frame, (nx, ny), dot_size, self._get_random_color(), -1)

    def process_frame(self, frame: np.ndarray, tracking_data: TrackingData = None) -> np.ndarray:
        h, w = frame.shape[:2]
        self.time_offset += 0.1 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # We calculate optical flow at half resolution to maintain high FPS
        small_gray = cv2.resize(gray, (w // 2, h // 2))

        # --- Initialize Maps & Vectorized Grid on First Frame ---
        if self.prev_gray is None or self.prev_gray.shape != small_gray.shape:
            self.prev_gray = small_gray.copy()
            self.flow_map = np.zeros((h, w, 3), dtype=np.float32)
            
            # Pre-render the dense dot mask once. This saves massive CPU time.
            self.bg_dot_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(0, h, self.bg_step):
                for x in range(0, w, self.bg_step):
                    cv2.circle(self.bg_dot_mask, (x, y), 1, (1, 1, 1), -1)

        # ==========================================
        # EXTRACT MASK EARLY (Used for both BG and FG)
        # ==========================================
        mask = None
        if tracking_data is not None and tracking_data.has_person:
            if tracking_data.segmentation_mask is not None:
                mask = tracking_data.segmentation_mask
                if len(mask.shape) > 2: mask = mask.squeeze()
                if mask.shape != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # ==========================================
        # 1. BACKGROUND: FLAME FLOW FADE EFFECT
        # ==========================================
        
        # Calculate dense optical flow (movement vectors)
        flow_small = cv2.calcOpticalFlowFarneback(self.prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = small_gray.copy()
        flow = cv2.resize(flow_small, (w, h)) * 2.0 

        # Map movement direction to Hue, and speed to Brightness
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Threshold the magnitude to remove camera noise/micro-jitter
        mag[mag < 1.0] = 0 
        
        hsv = np.zeros((h, w, 3), dtype=np.float32)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255.0
        hsv[..., 2] = np.clip(mag * 25, 0, 255) # Amplify speed for vibrant colors
        motion_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # Apply the segmentation mask so new flow colors ONLY spawn on the body
        if mask is not None:
            # Create a boolean mask and expand dimensions to match (h, w, 3)
            body_mask = (mask > 128).astype(np.float32)[..., np.newaxis]
            motion_bgr *= body_mask

        # Fluid Advection: Push the existing colors in the direction of the flow
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        new_x = x_coords - flow[..., 0]
        new_y = y_coords - flow[..., 1] - 1.5
        
        advected_map = cv2.remap(self.flow_map, new_x, new_y, interpolation=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # Add new movement colors to the advected map, fade it, and blur (diffuse)
        self.flow_map = advected_map + motion_bgr
        self.flow_map *= 0.90  # Decay factor (Fade out)
        self.flow_map = cv2.GaussianBlur(self.flow_map, (5, 5), 0)
        np.clip(self.flow_map, 0, 255, out=self.flow_map)

        # Render Background: Base faint greyscale dots + the vibrant flame flow
        base_gray = np.full((h, w, 3), 40, dtype=np.float32)
        final_bg_colors = np.clip(base_gray + self.flow_map, 0, 255).astype(np.uint8)
        
        output = final_bg_colors * self.bg_dot_mask

        # ==========================================
        # 2. FOREGROUND: VECTORIZED 3D DISPLACEMENT
        # ==========================================
        if mask is not None:
            # Create a coordinate grid based on the step size
            grid_y, grid_x = np.mgrid[0:h:self.step, 0:w:self.step]
            
            # Find which points in our grid fall inside the person mask
            is_person = mask[grid_y, grid_x] > 128
            
            # Extract only the active X and Y coordinates
            active_y = grid_y[is_person]
            active_x = grid_x[is_person]
            
            if len(active_y) > 0:
                # Grab intensities for all active points at once
                intensities = gray[active_y, active_x].astype(np.float32)
                
                # Calculate all displacements simultaneously
                displacements = (intensities / 255.0) * self.max_displacement
                
                # Calculate all new coordinates simultaneously
                nx = (active_x - displacements * 0.5).astype(np.int32)
                ny = (active_y - displacements).astype(np.int32)
                
                # Calculate all dot radii simultaneously
                radii = (1 + (intensities / 255.0) * 2).astype(np.int32)
                
                # Draw the dots (we still loop for the drawing function, but we've skipped processing empty space)
                for i in range(len(nx)):
                    cv2.circle(output, (nx[i], ny[i]), radii[i], self._get_random_color(), -1)

        # High-Density Face & Hands
        if tracking_data is not None and tracking_data.has_person:
            # Subsample the face landmarks (take every 5th point) to reduce clutter
            face_lms = tracking_data.face_landmarks[::5] if tracking_data.face_landmarks is not None else None
            
            self._draw_landmark_dots(output, face_lms, gray, h, w, 
                                     max_offset=self.max_displacement, dot_size=1)
            self._draw_landmark_dots(output, tracking_data.left_hand_landmarks, gray, h, w, 
                                     max_offset=self.max_displacement, dot_size=2)
            self._draw_landmark_dots(output, tracking_data.right_hand_landmarks, gray, h, w, 
                                     max_offset=self.max_displacement, dot_size=2)

        return output

    def close(self):
        pass

if __name__ == "__main__":
    print("Run this via main.py to pass the MediaPipe tracking_data into the processor.")