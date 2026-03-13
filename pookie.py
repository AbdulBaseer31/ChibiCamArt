import cv2
import numpy as np

class RibbonEffect:
    def __init__(self):
        self.enabled = True
        # Base Cute Pink (BGR)
        self.color = (180, 105, 255) 

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[RibbonEffect] Status: {'ON' if self.enabled else 'OFF'}")

    def draw(self, frame, tracking_data):
        if not self.enabled or not tracking_data or not tracking_data.has_person:
            return frame

        h, w = frame.shape[:2]
        
        # Use FACE landmarks (468 points) for absolute precision
        if tracking_data.face_landmarks is not None and len(tracking_data.face_landmarks) > 263:
            # Index 103 is the top-left area of the forehead
            anchor = tracking_data.face_landmarks[103]
            
            # Index 33 (outer left eye) and 263 (outer right eye) for face scale
            eye_l = tracking_data.face_landmarks[33]
            eye_r = tracking_data.face_landmarks[263]

            # Calculate head scale based on eye distance
            face_width = np.linalg.norm(
                np.array([eye_l[0], eye_l[1]]) - np.array([eye_r[0], eye_r[1]])
            )
            
            # Map normalized coordinates to pixel space
            center_x = int(anchor[0] * w)
            center_y = int(anchor[1] * h)
            
            # Scale ribbon relative to face width
            size = int(face_width * w * 0.45)
            
            if size > 5:
                self._draw_ribbon_shape(frame, (center_x, center_y), size)
        
        return frame

    def _draw_ribbon_shape(self, img, pos, s):
        """Draws a highly detailed 🎀 emoji style ribbon."""
        x, y = pos
        c = self.color
        
        # Darker pink for the inner creases of the bow
        dark_c = (int(c[0]*0.6), int(c[1]*0.6), int(c[2]*0.8)) 
        outline = (255, 255, 255) # White outline for pop
        
        # Dynamic line thickness based on distance from camera
        thick = max(1, s // 12)

        # ==========================================
        # 1. TAILS (Drawn first so they are in the back)
        # ==========================================
        # Left tail with "swallowtail" V-cut
        pts_tail_l = np.array([
            [x, y], 
            [x - int(s*0.7), y + int(s*1.2)], 
            [x - int(s*0.4), y + int(s*1.0)], 
            [x - int(s*0.1), y + int(s*1.3)], 
            [x, y + int(s*0.2)]
        ], np.int32)
        
        # Right tail with "swallowtail" V-cut
        pts_tail_r = np.array([
            [x, y], 
            [x + int(s*0.7), y + int(s*1.2)], 
            [x + int(s*0.4), y + int(s*1.0)], 
            [x + int(s*0.1), y + int(s*1.3)], 
            [x, y + int(s*0.2)]
        ], np.int32)

        # Draw tails & outlines
        cv2.fillPoly(img, [pts_tail_l, pts_tail_r], c)
        cv2.polylines(img, [pts_tail_l, pts_tail_r], True, outline, thick, cv2.LINE_AA)

        # ==========================================
        # 2. LOOPS (Large angled ellipses)
        # ==========================================
        # Left loop (angled up and left at -20 degrees)
        cv2.ellipse(img, (x - int(s*0.45), y), (int(s*0.5), int(s*0.35)), -20, 0, 360, c, -1)
        cv2.ellipse(img, (x - int(s*0.45), y), (int(s*0.5), int(s*0.35)), -20, 0, 360, outline, thick, cv2.LINE_AA)
        
        # Right loop (angled up and right at 20 degrees)
        cv2.ellipse(img, (x + int(s*0.45), y), (int(s*0.5), int(s*0.35)), 20, 0, 360, c, -1)
        cv2.ellipse(img, (x + int(s*0.45), y), (int(s*0.5), int(s*0.35)), 20, 0, 360, outline, thick, cv2.LINE_AA)

        # ==========================================
        # 3. CREASES (Dark inner voids to simulate fabric folds)
        # ==========================================
        cv2.ellipse(img, (x - int(s*0.5), y), (int(s*0.25), int(s*0.1)), -15, 0, 360, dark_c, -1)
        cv2.ellipse(img, (x + int(s*0.5), y), (int(s*0.25), int(s*0.1)), 15, 0, 360, dark_c, -1)

        # ==========================================
        # 4. KNOT (Drawn last so it overlaps the loops)
        # ==========================================
        cv2.ellipse(img, (x, y), (int(s*0.25), int(s*0.3)), 0, 0, 360, c, -1)
        cv2.ellipse(img, (x, y), (int(s*0.25), int(s*0.3)), 0, 0, 360, outline, thick, cv2.LINE_AA)