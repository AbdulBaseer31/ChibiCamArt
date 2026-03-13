"""
tracker.py (Tasks API Version - Enhanced with Clap-Explosion)

Modern pose tracking module using the MediaPipe Tasks API.
Provides 543 landmarks, Velocity detection, and an explosive clap gesture.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class TrackingData:
    has_person: bool = False
    pose_landmarks: Optional[np.ndarray] = None
    pose_visibility: Optional[np.ndarray] = None
    face_landmarks: Optional[np.ndarray] = None
    left_hand_landmarks: Optional[np.ndarray] = None
    right_hand_landmarks: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    
    # Legacy/Compatibility fields
    landmarks: Optional[np.ndarray] = None
    visibility: Optional[np.ndarray] = None
    full_543_landmarks: Optional[np.ndarray] = None
    
    # Speed & Effects
    velocity_color: Tuple[int, int, int] = (255, 0, 0) # BGR
    gesture: Optional[str] = None
    snap_active: bool = False
    confetti_points: List[List[float]] = None # [x, y, vx, vy]

class PoseTracker:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pose_model_path: str = "pose_landmarker_full.task",
        face_model_path: str = "face_landmarker.task",
        hand_model_path: str = "hand_landmarker.task"
    ) -> None:
        self.static_image_mode = static_image_mode
        self.enable_segmentation = enable_segmentation
        
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO

        # Initialize MediaPipe Tasks
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model_path),
                running_mode=running_mode, num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=enable_segmentation
            ))
        
        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_model_path),
                running_mode=running_mode, num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            ))
            
        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=running_mode, num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            ))
        
        # State Management
        self._prev_landmarks = None
        self._prev_time = time.time()
        self._frame_count = 0
        self._confetti_list = [] 
        self.dense_display = True

        # Explosion State
        self._explosion_active = False
        self._explosion_time = 0.0
        self._exploded_particles = [] # Format: [x, y, vx, vy, color]

        print(f"[PoseTracker] 543-Landmark Tasks API Ready.")

    def process_frame(self, frame: np.ndarray, return_visualization: bool = False, black_background: bool = False) -> Tuple[TrackingData, Optional[np.ndarray]]:
        self._frame_count += 1
        curr_time = time.time()
        dt = curr_time - self._prev_time
        timestamp_ms = int(self._frame_count * 1000 / 30)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Multi-model Inference
        if self.static_image_mode:
            p_res = self.pose_landmarker.detect(mp_image)
            f_res = self.face_landmarker.detect(mp_image)
            h_res = self.hand_landmarker.detect(mp_image)
        else:
            p_res = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            f_res = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            h_res = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        data = self._extract_tracking_data(p_res, f_res, h_res, frame.shape)
        
        # Logic: Velocity, Gestures, and Explosion Physics
        if data.has_person:
            data.velocity_color = self._update_velocity(data.pose_landmarks, dt)
            self._recognize_gestures(data, frame.shape)
        
        # Always update explosion physics if active, even if person is missing
        self._update_explosion_physics(frame.shape)
        
        self._prev_time = curr_time
        
        vis_frame = None
        if return_visualization:
            vis_frame = np.zeros_like(frame) if black_background else frame.copy()
            vis_frame = self._draw_landmarks_manual(vis_frame, data)
        
        return data, vis_frame

    def _extract_tracking_data(self, p_res, f_res, h_res, shape) -> TrackingData:
        data = TrackingData()
        
        # 1. Pose Data (33)
        if p_res and p_res.pose_landmarks:
            data.has_person = True
            lms = p_res.pose_landmarks[0]
            data.pose_landmarks = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
            data.pose_visibility = np.array([l.visibility for l in lms], dtype=np.float32)
            data.landmarks, data.visibility = data.pose_landmarks, data.pose_visibility
            if p_res.segmentation_masks:
                data.segmentation_mask = (p_res.segmentation_masks[0].numpy_view() > 0.5).astype(np.uint8) * 255

        # 2. Face Data (468)
        if f_res and f_res.face_landmarks:
            data.face_landmarks = np.array([[l.x, l.y, l.z] for l in f_res.face_landmarks[0]], dtype=np.float32)

        # 3. Hand Data (21 per hand)
        if h_res and h_res.hand_landmarks:
            for i, hand in enumerate(h_res.hand_landmarks):
                side = h_res.handedness[i][0].category_name
                arr = np.array([[l.x, l.y, l.z] for l in hand], dtype=np.float32)
                if side == "Left": data.left_hand_landmarks = arr
                else: data.right_hand_landmarks = arr

        # 4. Concatenate for Full 543 Landmark Set
        pose = data.pose_landmarks if data.pose_landmarks is not None else np.zeros((33, 3))
        face = data.face_landmarks if data.face_landmarks is not None else np.zeros((468, 3))
        lh = data.left_hand_landmarks if data.left_hand_landmarks is not None else np.zeros((21, 3))
        rh = data.right_hand_landmarks if data.right_hand_landmarks is not None else np.zeros((21, 3))
        
        data.full_543_landmarks = np.concatenate([pose, face, lh, rh], axis=0)
        return data

    def _update_velocity(self, current_lms, dt):
        if self._prev_landmarks is None or dt <= 0:
            self._prev_landmarks = current_lms
            return (255, 0, 0)
        
        speed = np.mean(np.linalg.norm(current_lms - self._prev_landmarks, axis=1)) / dt
        self._prev_landmarks = current_lms
        
        intensity = min(speed / 1.5, 1.0)
        if intensity < 0.5:
            return (255, int(510 * intensity), 0)
        return (int(255 - 510 * (intensity - 0.5)), 255, 255)

    def _recognize_gestures(self, data, shape):
        h, w = shape[:2]
        curr_time = time.time()

        # Cooldown Check: If exploded, wait 5 seconds before allowing logic to resume
        if self._explosion_active:
            if curr_time - self._explosion_time > 5.0:
                self._explosion_active = False
                self._exploded_particles = []
            return

        # 1. Snap Logic
        dist_l, dist_r = float('inf'), float('inf')
        thumb_l, thumb_r = None, None
        
        if data.left_hand_landmarks is not None:
            thumb_l, index_l = data.left_hand_landmarks[4], data.left_hand_landmarks[8]
            dist_l = np.linalg.norm(thumb_l - index_l)
            
        if data.right_hand_landmarks is not None:
            thumb_r, index_r = data.right_hand_landmarks[4], data.right_hand_landmarks[8]
            dist_r = np.linalg.norm(thumb_r - index_r)

        if dist_l < 0.05 and dist_r < 0.05 and not data.snap_active:
            data.snap_active = True
            for _ in range(20):
                self._confetti_list.append([int(thumb_l[0]*w), int(thumb_l[1]*h), np.random.uniform(-6,6), np.random.uniform(-12,-4)])
                self._confetti_list.append([int(thumb_r[0]*w), int(thumb_r[1]*h), np.random.uniform(-6,6), np.random.uniform(-12,-4)])
        elif dist_l > 0.12 or dist_r > 0.12:
            data.snap_active = False

        # 2. Clap/Explosion Logic
        if data.left_hand_landmarks is not None and data.right_hand_landmarks is not None:
            # Distance between palm centers (Landmark 9)
            palm_l, palm_r = data.left_hand_landmarks[9], data.right_hand_landmarks[9]
            if np.linalg.norm(palm_l - palm_r) < 0.07:
                self._explosion_active = True
                self._explosion_time = curr_time
                # Convert all 543 landmarks to physics particles
                for lm in data.full_543_landmarks:
                    self._exploded_particles.append([
                        lm[0] * w, lm[1] * h, 
                        np.random.uniform(-18, 18), np.random.uniform(-18, 18),
                        (np.random.randint(100,255), np.random.randint(100,255), 255)
                    ])

        # Confetti update
        for p in self._confetti_list:
            p[0] += p[2]; p[1] += p[3]; p[3] += 0.6 
        self._confetti_list = [p for p in self._confetti_list if p[1] < h]
        data.confetti_points = self._confetti_list

    def _update_explosion_physics(self, shape):
        h = shape[0]
        for p in self._exploded_particles:
            p[0] += p[2] # vx
            p[1] += p[3] # vy
            p[3] += 0.8  # gravity
        # Keep particles until they fall off screen
        self._exploded_particles = [p for p in self._exploded_particles if p[1] < h]

    def _draw_landmarks_manual(self, frame, data):
        h, w = frame.shape[:2]

        # If exploded, draw particles and skip regular rendering
        if self._explosion_active:
            for x, y, vx, vy, color in self._exploded_particles:
                cv2.circle(frame, (int(x), int(y)), 2, color, -1)
            return frame

        if not data.has_person: return frame
        
        # 1. Confetti
        if data.confetti_points:
            for x, y, vx, vy in data.confetti_points:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
        
        # 2. Dense Face Mesh
        if self.dense_display and data.face_landmarks is not None:
            for lm in data.face_landmarks:
                cv2.circle(frame, (int(lm[0]*w), int(lm[1]*h)), 1, (0, 255, 0), -1)

        # 3. Dense Hands
        if self.dense_display:
            for hand in [data.left_hand_landmarks, data.right_hand_landmarks]:
                if hand is not None:
                    for lm in hand:
                        cv2.circle(frame, (int(lm[0]*w), int(lm[1]*h)), 2, (0, 255, 255), -1)

        # 4. Pose Skeleton
        color = data.velocity_color
        conns = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16)]
        for s, e in conns:
            if data.pose_visibility[s] > 0.5 and data.pose_visibility[e] > 0.5:
                p1 = (int(data.pose_landmarks[s][0]*w), int(data.pose_landmarks[s][1]*h))
                p2 = (int(data.pose_landmarks[e][0]*w), int(data.pose_landmarks[e][1]*h))
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        
        return frame

    def close(self):
        self.pose_landmarker.close()
        self.face_landmarker.close()
        self.hand_landmarker.close()

    def __enter__(self): return self
    def __exit__(self, et, ev, tb): self.close()