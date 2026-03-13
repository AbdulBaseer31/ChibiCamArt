import cv2
import numpy as np
import random
from old_tracker import TrackingData

class HologramEffect:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Strict Blue-Only Palette (BGR format)
        self.void_blue = (40, 20, 0)        # Deep dark navy (Replaces Black)
        self.panel_bg_blue = (80, 40, 0)    # Slightly lighter navy for panel backgrounds
        self.holo_cyan = (255, 255, 0)      # Bright Cyan
        self.sky_blue = (255, 200, 0)       # Sky Blue
        self.pale_cyan = (255, 255, 150)    # Almost-white pale blue for text readability
        self.teal = (200, 150, 0)           # Mid-tone teal
        
        self.pulse_time = 0.0

    def _draw_holographic_body(self, holo_layer, mask, h, w):
        # Find the edges of the human silhouette
        edges = cv2.Canny(mask, 50, 150)
        holo_layer[edges > 0] = self.holo_cyan
        
        # Add horizontal scanlines
        step = 6
        scanline_mask = np.zeros_like(mask)
        scanline_mask[::step, :] = 255
        
        internal_lines = cv2.bitwise_and(mask, scanline_mask)
        holo_layer[internal_lines > 0] = self.teal

        # Apply random horizontal glitch tearing ONLY to the body
        if random.random() < 0.4:
            for _ in range(random.randint(2, 6)):
                y_start = random.randint(0, h - 20)
                
                # Clamp y_end to never exceed the frame height
                y_end = min(h, y_start + random.randint(5, 30))
                actual_height = y_end - y_start
                
                shift = random.randint(-40, 40)
                if shift != 0 and actual_height > 0:
                    M = np.float32([[1, 0, shift], [0, 1, 0]])
                    holo_layer[y_start:y_end] = cv2.warpAffine(
                        holo_layer[y_start:y_end], M, (w, actual_height)
                    )

    def _draw_simplistic_face(self, target_img, face_lms, h, w):
        if face_lms is None: return
        
        jaw_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        jaw_pts = []
        for idx in jaw_indices:
            if idx < len(face_lms):
                px = max(0, min(int(face_lms[idx][0] * w), w - 1))
                py = max(0, min(int(face_lms[idx][1] * h), h - 1))
                jaw_pts.append([px, py])
                
        if jaw_pts:
            pts = np.array(jaw_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(target_img, [pts], isClosed=True, color=self.sky_blue, thickness=1)

        left_eye = [33, 133]
        right_eye = [362, 263]
        for eye in [left_eye, right_eye]:
            if max(eye) < len(face_lms):
                p1 = (int(face_lms[eye[0]][0] * w), int(face_lms[eye[0]][1] * h))
                p2 = (int(face_lms[eye[1]][0] * w), int(face_lms[eye[1]][1] * h))
                cv2.line(target_img, p1, p2, self.pale_cyan, 2)

        mouth_corners = [78, 308]
        if max(mouth_corners) < len(face_lms):
            m1 = (int(face_lms[mouth_corners[0]][0] * w), int(face_lms[mouth_corners[0]][1] * h))
            m2 = (int(face_lms[mouth_corners[1]][0] * w), int(face_lms[mouth_corners[1]][1] * h))
            cv2.line(target_img, m1, m2, self.sky_blue, 2)

    def _draw_hand_panel(self, target_img, hand_lms, h, w, side="RIGHT"):
        if hand_lms is None: return
        
        palm_center = hand_lms[9]
        cx = int(palm_center[0] * w)
        cy = int(palm_center[1] * h)
        
        panel_w, panel_h = 160, 100
        offset_x = 40 if side == "LEFT" else -panel_w - 40
        top_left = (cx + offset_x, cy - panel_h // 2)
        bottom_right = (top_left[0] + panel_w, top_left[1] + panel_h)
        
        # Localized transparency for just the panel background
        overlay = target_img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, self.panel_bg_blue, -1)
        # Blend the panel background back onto the target image
        cv2.addWeighted(overlay, 0.75, target_img, 0.25, 0, target_img)
        
        # Draw solid, non-flickering borders and text
        cv2.rectangle(target_img, top_left, bottom_right, self.sky_blue, 1)
        
        text_x = top_left[0] + 10
        text_y = top_left[1] + 25
        
        sys_temp = f"SYS.TEMP: {random.randint(45, 80)}C"
        mem_usage = f"MEM.ALLOC: {random.randint(1024, 4096)}MB"
        status = "UPLINK: ACTIVE" if int(self.pulse_time * 10) % 2 == 0 else "UPLINK: SYNC..."
        
        cv2.putText(target_img, f"[{side} PANEL]", (text_x, text_y), self.font, 0.4, self.pale_cyan, 1)
        cv2.putText(target_img, sys_temp, (text_x, text_y + 25), self.font, 0.35, self.sky_blue, 1)
        cv2.putText(target_img, mem_usage, (text_x, text_y + 45), self.font, 0.35, self.sky_blue, 1)
        cv2.putText(target_img, status, (text_x, text_y + 65), self.font, 0.35, self.holo_cyan, 1)

    def process_frame(self, frame: np.ndarray, tracking_data: TrackingData = None) -> np.ndarray:
        h, w = frame.shape[:2]
        self.pulse_time += 0.1
        
        # 1. Start with the actual real-world background
        output = frame.copy()

        if tracking_data is not None and tracking_data.has_person:
            
            # --- A. Flickering Body Hologram ---
            if tracking_data.segmentation_mask is not None:
                mask = tracking_data.segmentation_mask
                if len(mask.shape) > 2: mask = mask.squeeze()
                if mask.shape != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                mask_bool = mask > 128
                
                # Replace the physical person with a deep navy void instead of black
                output[mask_bool] = self.void_blue
                
                # Draw the glitching body on a temporary black canvas (so additive blending works)
                holo_layer = np.zeros_like(output)
                
                # Pass a standard 255 uint8 mask for edge detection
                self._draw_holographic_body(holo_layer, mask_bool.astype(np.uint8) * 255, h, w)
                
                # Randomize opacity for the flickering effect
                flicker_alpha = random.uniform(0.1, 0.4)
                if random.random() < 0.1: flicker_alpha = random.uniform(0.6, 0.9)
                
                # Blend ONLY the body using the flicker opacity
                output = cv2.addWeighted(holo_layer, flicker_alpha, output, 1.0, 0)

            # --- B. Stable UI and Face ---
            # By drawing these directly onto 'output' AFTER the body is blended, 
            # the UI remains 100% stable and does not flicker or glitch.
            self._draw_simplistic_face(output, tracking_data.face_landmarks, h, w)
            self._draw_hand_panel(output, tracking_data.left_hand_landmarks, h, w, "RIGHT")
            self._draw_hand_panel(output, tracking_data.right_hand_landmarks, h, w, "LEFT")

        return output

    def close(self):
        pass

if __name__ == "__main__":
    print("Run this via main.py to pass the MediaPipe tracking_data into the processor.")