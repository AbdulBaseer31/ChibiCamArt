import cv2
import numpy as np
import time
import random
import os
import pygame  # ADDED: For concurrent audio playback
from dataclasses import dataclass
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import resource_path

# ==========================================
# 1. DATA STRUCTURE
# ==========================================
@dataclass
class TrackingData:
    has_person: bool = False
    face_landmarks: Optional[np.ndarray] = None
    left_hand_landmarks: Optional[np.ndarray] = None
    right_hand_landmarks: Optional[np.ndarray] = None
    left_gesture: str = "None"
    right_gesture: str = "None"
    segmentation_mask: Optional[np.ndarray] = None

# ==========================================
# 2. RENDERER (Dots + Mesh + Physics + Trophy + Thumbs)
# ==========================================
class SimpleEffectRenderer:
    def __init__(self):
        self.step = 8                
        self.max_displacement = 30    
        
        # Explosion Physics State
        self.explosion_end_time = 0.0
        self.explosion_particles = [] 
        
        # SALT PARTICLES STATE 
        self.salt_particles = [] 

        # VICTORY TROPHY & CONFETTI STATE
        self.victory_start_time = {'left': None, 'right': None}
        self.trophy_end_time = 0.0
        self.confetti_particles = [] 
        self.trophy_bgra = None      
        
        # THUMBS UP / DOWN STATE
        self.thumb_hold_start = {'left': None, 'right': None}
        self.thumb_current_gesture = {'left': None, 'right': None}
        self.thumb_cooldown_end = {'Thumb_Up': 0.0, 'Thumb_Down': 0.0}
        self.thumb_display_end = {'Thumb_Up': 0.0, 'Thumb_Down': 0.0}
        
        self.img_thumb_up = None
        self.img_thumb_down = None
        self.snd_cheers = None
        self.snd_boos = None

        self.props_loaded = False

    def load_png_prop(self, filename, target_size=(300, 300)):
        """Generic PNG loader replacing the hardcoded trophy loader"""
        png_path = resource_path(os.path.join("props", filename))
        if not os.path.exists(png_path):
            print(f"[Warning] Prop PNG not found at {png_path}!")
            return None
            
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[Error] Failed to load PNG at {png_path}")
            return None
            
        return cv2.resize(img, target_size)

    def trigger_victory_celebration(self, curr_time, w):
        self.trophy_end_time = curr_time + 4.0  
        self.confetti_particles = []
        
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        
        for _ in range(150):
            is_left = random.choice([True, False])
            x_start = random.uniform(0, w * 0.2) if is_left else random.uniform(w * 0.8, w)
            vx = random.uniform(2, 10) if is_left else random.uniform(-10, -2)
            
            self.confetti_particles.append({
                'x': x_start,
                'y': random.uniform(-100, 0),
                'vx': vx,
                'vy': random.uniform(5, 15),
                'color': random.choice(colors),
                'size': random.randint(4, 10),
                'angle': random.uniform(0, 360),
                'spin': random.uniform(-10, 10)
            })

    def overlay_bgra(self, background, overlay, x, y):
        """Vectorized BGRA overlay for instant blending"""
        h_bg, w_bg = background.shape[:2]
        h_ol, w_ol = overlay.shape[:2]

        if x >= w_bg or y >= h_bg or x + w_ol <= 0 or y + h_ol <= 0:
            return background

        x1, x2 = max(0, x), min(w_bg, x + w_ol)
        y1, y2 = max(0, y), min(h_bg, y + h_ol)
        
        ol_x1, ol_x2 = max(0, -x), min(w_ol, w_bg - x)
        ol_y1, ol_y2 = max(0, -y), min(h_ol, h_bg - y)

        overlay_crop = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
        
        # Vectorized Alpha Extraction & Blending
        alpha = (overlay_crop[:, :, 3] / 255.0)[:, :, np.newaxis]
        background[y1:y2, x1:x2] = (alpha * overlay_crop[:, :, :3] + 
                                    (1 - alpha) * background[y1:y2, x1:x2]).astype(np.uint8)
        return background

    def render(self, frame: np.ndarray, data: TrackingData) -> np.ndarray:
        h, w = frame.shape[:2]
        curr_time = time.time()
        
        output = np.zeros((h, w, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # LAZY LOAD PNGS & AUDIO
        if not self.props_loaded:
            self.trophy_bgra = self.load_png_prop("trophy-svgrepo-com.png", target_size=(int(h * 0.4), int(h * 0.4)))
            self.img_thumb_up = self.load_png_prop("Thumbsup.png", target_size=(int(h * 0.3), int(h * 0.3)))
            self.img_thumb_down = self.load_png_prop("Thumbsdown.png", target_size=(int(h * 0.3), int(h * 0.3)))
            
            try:
                pygame.mixer.init()
                self.snd_cheers = pygame.mixer.Sound(resource_path(os.path.join("props", "Cheers.ogg")))
                self.snd_boos = pygame.mixer.Sound(resource_path(os.path.join("props", "Boos.ogg")))
            except Exception as e:
                print(f"[Audio Error] Check if Cheers.ogg and Boos.ogg exist in the props folder: {e}")
                
            self.props_loaded = True

        # UPDATE SALT PARTICLES PHYSICS
        active_salt = []
        for p in self.salt_particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 1.2  
            
            cv2.circle(output, (int(p['x']), int(p['y'])), p['r'], p['color'], -1)
            
            if p['y'] < h:
                active_salt.append(p)
        self.salt_particles = active_salt

        # UPDATE EXPLOSION PHYSICS
        if curr_time < self.explosion_end_time:
            for p in self.explosion_particles:
                p[0] += p[3]
                p[1] += p[4]
                p[4] += 0.8
                cv2.circle(output, (int(p[0]), int(p[1])), int(p[2]), p[5], -1) 
            return output
        if self.explosion_particles:
            self.explosion_particles = []

        # ==========================================
        # NORMAL RENDERING 
        # ==========================================
        current_frame_dots = [] 
        
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        if data.face_landmarks is not None:
            pts = np.array([[int(lm[0]*w), int(lm[1]*h)] for lm in data.face_landmarks], dtype=np.int32)
            cv2.fillConvexPoly(exclusion_mask, cv2.convexHull(pts), 255)
            
        for hand in [data.left_hand_landmarks, data.right_hand_landmarks]:
            if hand is not None:
                pts = np.array([[int(lm[0]*w), int(lm[1]*h)] for lm in hand], dtype=np.int32)
                cv2.fillConvexPoly(exclusion_mask, cv2.convexHull(pts), 255)
                
        exclusion_mask = cv2.dilate(exclusion_mask, np.ones((15, 15), np.uint8), iterations=1)

        if data.segmentation_mask is not None:
            mask = data.segmentation_mask
            if len(mask.shape) > 2: mask = mask.squeeze()
            if mask.shape != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            grid_y, grid_x = np.mgrid[0:h:self.step, 0:w:self.step]
            is_person = (mask[grid_y, grid_x] > 128) & (exclusion_mask[grid_y, grid_x] == 0)
            active_y, active_x = grid_y[is_person], grid_x[is_person]
            
            if len(active_y) > 0:
                intensities = gray[active_y, active_x].astype(np.float32)
                displacements = (intensities / 255.0) * self.max_displacement
                nx = (active_x - displacements * 0.5).astype(np.int32)
                ny = (active_y - displacements).astype(np.int32)
                radii = (1 + (intensities / 255.0) * 2).astype(np.int32)
                for i in range(len(nx)):
                    current_frame_dots.append((nx[i], ny[i], radii[i], (255, 255, 255)))

        if data.face_landmarks is not None:
            for lm in data.face_landmarks:
                current_frame_dots.append((int(lm[0]*w), int(lm[1]*h), 1, (0, 255, 0)))

        for hand in [data.left_hand_landmarks, data.right_hand_landmarks]:
            if hand is not None:
                for lm in hand:
                    current_frame_dots.append((int(lm[0]*w), int(lm[1]*h), 2, (0, 255, 0)))

        # ==========================================
        # GESTURE LOGIC 
        # ==========================================
        
        # CLAP CHECK 
        if data.left_hand_landmarks is not None and data.right_hand_landmarks is not None:
            palm_l, palm_r = data.left_hand_landmarks[9], data.right_hand_landmarks[9]
            if np.linalg.norm(palm_l - palm_r) < 0.08:
                self.explosion_end_time = curr_time + 5.0
                for x, y, r, _ in current_frame_dots:
                    self.explosion_particles.append([x, y, r, random.uniform(-18, 18), random.uniform(-18, 4), (0, random.randint(0, 255), 255)])
                return output

        # GESTURE ITERATION
        for hand, gesture, side in [(data.left_hand_landmarks, data.left_gesture, 'left'), 
                                    (data.right_hand_landmarks, data.right_gesture, 'right')]:
            
            # 1. Victory Logic
            if gesture == "Victory":
                if self.victory_start_time[side] is None:
                    self.victory_start_time[side] = curr_time
                elif curr_time - self.victory_start_time[side] >= 2.0:
                    if curr_time > self.trophy_end_time: 
                        self.trigger_victory_celebration(curr_time, w)
            else:
                self.victory_start_time[side] = None 
                
            # 2. Pinch Logic (Salt Flow)
            if hand is not None:
                thumb_tip, index_tip = hand[4], hand[8]
                middle_tip, ring_tip, pinky_tip = hand[12], hand[16], hand[20]
                
                pinch_dist = np.linalg.norm(thumb_tip - index_tip)
                middle_dist = np.linalg.norm(middle_tip - thumb_tip)
                ring_dist = np.linalg.norm(ring_tip - thumb_tip)
                pinky_dist = np.linalg.norm(pinky_tip - thumb_tip)
                
                is_pinching = (pinch_dist < 0.04) and (middle_dist > 0.08) and (ring_dist > 0.08) and (pinky_dist > 0.08)
                
                if gesture != "Closed_Fist" and ("pinch" in gesture.lower() or is_pinching):
                    px, py = int(((thumb_tip[0] + index_tip[0]) / 2) * w), int(((thumb_tip[1] + index_tip[1]) / 2) * h)
                    
                    for _ in range(12):
                        self.salt_particles.append({
                            'x': float(px) + random.uniform(-4.0, 4.0),
                            'y': float(py),
                            'vx': random.uniform(-1.0, 1.0),
                            'vy': random.uniform(6.0, 14.0), 
                            'r': random.randint(1, 2),      
                            'color': (255, 255, 255) 
                        })

            # 3. Thumbs Up / Down Logic
            if gesture in ["Thumb_Up", "Thumb_Down"]:
                if curr_time > self.thumb_cooldown_end[gesture]:
                    if self.thumb_current_gesture[side] != gesture:
                        self.thumb_current_gesture[side] = gesture
                        self.thumb_hold_start[side] = curr_time
                    elif curr_time - self.thumb_hold_start[side] >= 1.5:
                        # TRIGGER!
                        self.thumb_display_end[gesture] = curr_time + 3.0 # Show image for 3 seconds
                        self.thumb_cooldown_end[gesture] = curr_time + 5.0 # 5 sec cooldown
                        
                        # Play appropriate audio
                        if gesture == "Thumb_Up" and self.snd_cheers:
                            self.snd_cheers.play()
                        elif gesture == "Thumb_Down" and self.snd_boos:
                            self.snd_boos.play()
                            
                        # Reset tracking so it doesn't fire every frame
                        self.thumb_hold_start[side] = None
                        self.thumb_current_gesture[side] = None
                else:
                    self.thumb_hold_start[side] = None
                    self.thumb_current_gesture[side] = None
            else:
                self.thumb_hold_start[side] = None
                self.thumb_current_gesture[side] = None


        # ==========================================
        # FINALIZE DRAWING
        # ==========================================
        for x, y, r, color in current_frame_dots:
            cv2.circle(output, (x, y), r, color, -1)

        # DRAW TROPHY & CONFETTI 
        if curr_time < self.trophy_end_time:
            # 1. Draw Confetti
            active_confetti = []
            for c in self.confetti_particles:
                c['x'] += c['vx']
                c['y'] += c['vy']
                c['vy'] += 0.2 
                c['angle'] += c['spin']
                
                if c['y'] < h:
                    rect = ((c['x'], c['y']), (c['size'], c['size']), c['angle'])
                    try:
                        box = cv2.boxPoints(rect)
                    except AttributeError:
                        box = cv2.cv.BoxPoints(rect)
                    box = np.int32(box)
                    cv2.fillPoly(output, [box], c['color'])
                    active_confetti.append(c)
            self.confetti_particles = active_confetti

            # 2. Draw PNG Trophy Overlay
            if self.trophy_bgra is not None:
                th, tw = self.trophy_bgra.shape[:2]
                tx, ty = (w - tw) // 2, (h - th) // 2
                output = self.overlay_bgra(output, self.trophy_bgra, tx, ty)

        # DRAW THUMBS UP
        if curr_time < self.thumb_display_end["Thumb_Up"] and self.img_thumb_up is not None:
            output = self.overlay_bgra(output, self.img_thumb_up, 50, 50) # Drawn on Top-Left

        # DRAW THUMBS DOWN
        if curr_time < self.thumb_display_end["Thumb_Down"] and self.img_thumb_down is not None:
            th, tw = self.img_thumb_down.shape[:2]
            output = self.overlay_bgra(output, self.img_thumb_down, w - tw - 50, 50) # Drawn on Top-Right

        return output

# ==========================================
# 3. TRACKER (Bridge for main.py)
# ==========================================
class PoseTracker:
    def __init__(
        self,
        pose_model_path: str = "pose_landmarker_full.task",
        face_model_path: str = "face_landmarker.task",
        gesture_model_path: str = "gesture_recognizer.task", 
        device: str = "cpu",
        **kwargs 
    ) -> None:
        pose_model_path = resource_path(pose_model_path)
        face_model_path = resource_path(face_model_path)
        gesture_model_path = resource_path(gesture_model_path)
        
        running_mode = vision.RunningMode.VIDEO
        
        # Determine Delegate (Forced CPU)
        delegate = python.BaseOptions.Delegate.CPU
        print(f"[MediaPipeTracker] Using delegate: {delegate.name}")

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model_path, delegate=delegate),
                running_mode=running_mode, num_poses=1,
                output_segmentation_masks=True
            ))
        
        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_model_path, delegate=delegate),
                running_mode=running_mode, num_faces=1
            ))
            
        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=gesture_model_path, delegate=delegate),
                running_mode=running_mode, num_hands=2
            ))
        
        self._frame_count = 0
        self.renderer = SimpleEffectRenderer()
        self.executor = ThreadPoolExecutor(max_workers=3)
        print("[PoseTracker] Initialized with Gesture Recognizer, Physics Support & Multithreading.")

    def process_frame(
        self, 
        frame: np.ndarray, 
        return_visualization: bool = False, 
        black_background: bool = False, 
        **kwargs
    ) -> Tuple[TrackingData, Optional[np.ndarray]]:
        
        self._frame_count += 1
        timestamp_ms = int(self._frame_count * 1000 / 30)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        future_pose = self.executor.submit(self.pose_landmarker.detect_for_video, mp_image, timestamp_ms)
        future_face = self.executor.submit(self.face_landmarker.detect_for_video, mp_image, timestamp_ms)
        future_gest = self.executor.submit(self.gesture_recognizer.recognize_for_video, mp_image, timestamp_ms)

        p_res = future_pose.result()
        f_res = future_face.result()
        g_res = future_gest.result()
        
        data = self._extract_tracking_data(p_res, f_res, g_res, frame.shape)
        
        vis_frame = None
        if return_visualization:
            vis_frame = self.renderer.render(frame, data)
            
        return data, vis_frame

    def _extract_tracking_data(self, p_res, f_res, g_res, shape) -> TrackingData:
        data = TrackingData()
        
        if p_res and p_res.pose_landmarks:
            data.has_person = True
            if p_res.segmentation_masks:
                data.segmentation_mask = (p_res.segmentation_masks[0].numpy_view() > 0.5).astype(np.uint8) * 255

        if f_res and f_res.face_landmarks:
            data.face_landmarks = np.array([[l.x, l.y, l.z] for l in f_res.face_landmarks[0]], dtype=np.float32)

        if g_res and g_res.hand_landmarks:
            for i, hand in enumerate(g_res.hand_landmarks):
                side = g_res.handedness[i][0].category_name
                arr = np.array([[l.x, l.y, l.z] for l in hand], dtype=np.float32)
                
                gesture = "None"
                if g_res.gestures and len(g_res.gestures) > i and g_res.gestures[i]:
                    gesture = g_res.gestures[i][0].category_name
                
                if side == "Left": 
                    data.left_hand_landmarks = arr
                    data.left_gesture = gesture
                else: 
                    data.right_hand_landmarks = arr
                    data.right_gesture = gesture

        return data

    def close(self):
        self.pose_landmarker.close()
        self.face_landmarker.close()
        self.gesture_recognizer.close()
        self.executor.shutdown(wait=True)
        # Quit PyGame audio safely
        try:
            pygame.mixer.quit()
        except:
            pass

    def __enter__(self): return self
    def __exit__(self, et, ev, tb): self.close()

# ==========================================
# 4. MAIN EXECUTION (Standalone Test)
# ==========================================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = PoseTracker()
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror display
        tracking_data, output_frame = tracker.process_frame(frame, return_visualization=True)
        
        if output_frame is not None:
            cv2.imshow("Simplistic Dot & Mesh Effect", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()