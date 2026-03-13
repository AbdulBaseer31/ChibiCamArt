import cv2
import numpy as np
import random
from old_tracker import TrackingData

class GlitchEffect:
    def __init__(self):
        # Randomized intensity state for foreground pulsing
        self.glitch_intensity = 0.0
        
        # State variable for the continuous background roll
        self.bg_scroll_offset = 0

    def process_frame(self, frame: np.ndarray, tracking_data: TrackingData = None) -> np.ndarray:
        h, w = frame.shape[:2]
        glitched_frame = frame.copy()

        # --- 1. GENERATE CONTINUOUS BACKGROUND GLITCH (BLUE/TEAL, BRIGHTER) ---
        # Generate chunky horizontal noise bands favoring Blue and Green
        band_width = max(1, w // 20)
        noise_b = np.random.randint(120, 255, (h, band_width), dtype=np.uint8)
        noise_g = np.random.randint(80, 220, (h, band_width), dtype=np.uint8)
        noise_r = np.random.randint(0, 90, (h, band_width), dtype=np.uint8)
        band_noise = cv2.merge([noise_b, noise_g, noise_r])
        band_noise = cv2.resize(band_noise, (w, h), interpolation=cv2.INTER_NEAREST)

        # Generate fine static favoring Blue and Green
        static_b = np.random.randint(100, 255, (h, w), dtype=np.uint8)
        static_g = np.random.randint(60, 180, (h, w), dtype=np.uint8)
        static_r = np.random.randint(0, 60, (h, w), dtype=np.uint8)
        static_noise = cv2.merge([static_b, static_g, static_r])
        
        # Blend them together for a rich texture
        bg_glitch = cv2.addWeighted(static_noise, 0.4, band_noise, 0.6, 0)

        # Create seamless, bright rolling tracking waves (no black lines)
        self.bg_scroll_offset = (self.bg_scroll_offset + 10) % h
        
        # We'll create 2 wide, glowing bands that roll down the screen
        for i in range(2):
            y = (self.bg_scroll_offset + i * (h // 2)) % h
            thickness = h // 6
            
            # Boost the brightness/cyan values in the tracking band
            if y + thickness <= h:
                tint = np.zeros_like(bg_glitch[y:y+thickness])
                tint[:, :, 0] = 60  # Boost Blue
                tint[:, :, 1] = 40  # Boost Green
                bg_glitch[y:y+thickness] = cv2.add(bg_glitch[y:y+thickness], tint)
            else:
                # Handle screen wrap-around seamlessly
                rem = (y + thickness) - h
                
                tint1 = np.zeros_like(bg_glitch[y:h])
                tint1[:, :, 0], tint1[:, :, 1] = 60, 40
                bg_glitch[y:h] = cv2.add(bg_glitch[y:h], tint1)
                
                tint2 = np.zeros_like(bg_glitch[0:rem])
                tint2[:, :, 0], tint2[:, :, 1] = 60, 40
                bg_glitch[0:rem] = cv2.add(bg_glitch[0:rem], tint2)

        # Randomize foreground glitch intensity
        if random.random() < 0.80: 
            self.glitch_intensity = random.uniform(0.4, 1.0)
        else:
            self.glitch_intensity = max(0.0, self.glitch_intensity - 0.15)

        # --- 2. APPLY MASKS AND FOREGROUND EFFECTS ---
        if tracking_data is not None and tracking_data.has_person:
            
            if tracking_data.segmentation_mask is not None:
                mask = tracking_data.segmentation_mask
                
                if len(mask.shape) > 2:
                    mask = mask.squeeze()

                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                mask_bool = mask > 128

                # Apply the continuous background glitch everywhere EXCEPT the person
                glitched_frame[~mask_bool] = bg_glitch[~mask_bool]

                # --- A. Silhouette Glitch ---
                if self.glitch_intensity > 0.1:
                    shift_amount = int(80 * self.glitch_intensity)
                    if shift_amount > 0:
                        fg_b = glitched_frame[:, :, 0]
                        fg_r = glitched_frame[:, :, 2]

                        M_r = np.float32([[1, 0, shift_amount], [0, 1, 0]])
                        M_b = np.float32([[1, 0, -shift_amount], [0, 1, 0]])
                        
                        shifted_r = cv2.warpAffine(fg_r, M_r, (w, h))
                        shifted_b = cv2.warpAffine(fg_b, M_b, (w, h))

                        glitched_frame[mask_bool, 0] = shifted_b[mask_bool].astype(np.uint8)
                        glitched_frame[mask_bool, 2] = shifted_r[mask_bool].astype(np.uint8)

                    num_slices = int(35 * self.glitch_intensity)
                    for _ in range(num_slices):
                        y_start = random.randint(0, h - 30)
                        slice_height = random.randint(5, 50)
                        y_end = min(h, y_start + slice_height)
                        
                        x_shift = random.randint(-40, 40)

                        if x_shift != 0:
                            M_slice = np.float32([[1, 0, x_shift], [0, 1, 0]])
                            band = glitched_frame[y_start:y_end, :]
                            shifted_band = cv2.warpAffine(band, M_slice, (w, y_end - y_start))
                            
                            slice_mask = mask_bool[y_start:y_end, :]
                            temp_slice = glitched_frame[y_start:y_end, :]
                            temp_slice[slice_mask] = shifted_band[slice_mask]
                            glitched_frame[y_start:y_end, :] = temp_slice

            # --- C. Hand Tracking Errors (Ghosting) ---
            def draw_ghost_hands(landmarks, color):
                if landmarks is not None and self.glitch_intensity > 0.2:
                    shift_x = random.randint(-80, 80)
                    shift_y = random.randint(-80, 80)
                    for lm in landmarks:
                        cx = int(lm[0] * w) + shift_x
                        cy = int(lm[1] * h) + shift_y
                        if 0 <= cx < w and 0 <= cy < h:
                            rect_size = random.randint(2, 8)
                            cv2.rectangle(glitched_frame, (cx, cy), (cx + rect_size, cy + rect_size), color, -1)

            draw_ghost_hands(tracking_data.left_hand_landmarks, (255, 0, 255))
            draw_ghost_hands(tracking_data.right_hand_landmarks, (255, 0, 255))

        else:
            # If no person is detected at all, show the full background glitch
            glitched_frame = bg_glitch

        return glitched_frame

    def close(self):
        pass

if __name__ == "__main__":
    print("Run this via main.py to pass the MediaPipe tracking_data into the processor.")