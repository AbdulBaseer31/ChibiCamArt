"""
tracker.py
Multi-person pose tracking using YOLOv8 with CUDA acceleration and 
visibility filtering to prevent ghost/hallucinated limbs.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from ultralytics import YOLO
from utils import resource_path

@dataclass
class TrackingData:
    has_person: bool = False
    landmarks: Optional[np.ndarray] = None
    all_poses: Optional[List[np.ndarray]] = None
    all_visibilities: Optional[List[np.ndarray]] = None
    visibility: Optional[np.ndarray] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None

class PoseTracker:
    def __init__(
        self,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        device: str = None
    ) -> None:
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self._frame_count = 0  # <--- FIXED: Added missing attribute
        
        # GPU Check - use provided device or auto-detect
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[PoseTracker] Using device: {self.device}")
        
        # Loading yolo26m-pose (Medium)
        print("[PoseTracker] Loading yolo26m-pose model...")
        self.model = YOLO(resource_path('yolo26m-pose.pt'))
        self.model.to(self.device)  # <--- FORCED CUDA
        
        # COCO Connections (17 keypoints)
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def process_frame(self, frame: np.ndarray, return_visualization: bool = False, black_background: bool = False) -> Tuple[TrackingData, Optional[np.ndarray]]:
        self._frame_count += 1
        
        # SPEED-HACKS: imgsz=320 and half=True significantly boost FPS on GPU
        results = self.model(
            frame, 
            conf=self.confidence, 
            iou=self.iou_threshold, 
            max_det=self.max_detections, 
            device=self.device,
            imgsz=320,      # Faster processing for webcam distance
            half=(self.device == 'cuda'), # FP16 speedup
            verbose=False
        )
        
        tracking_data = self._extract_tracking_data(results[0], frame.shape)
        
        vis_frame = None
        if return_visualization:
            vis_frame = np.zeros_like(frame) if black_background else frame.copy()
            vis_frame = self._draw_landmarks(vis_frame, tracking_data)
        
        return tracking_data, vis_frame

    def _extract_tracking_data(self, result: Any, frame_shape: Tuple[int, ...]) -> TrackingData:
        data = TrackingData()
        h, w = frame_shape[:2]

        if result.keypoints is not None and len(result.keypoints) > 0:
            data.has_person = True
            all_poses = []
            all_visibilities = []
            
            keypoints_data = result.keypoints.xy.cpu().numpy()
            conf_data = result.keypoints.conf.cpu().numpy()

            for person_idx in range(keypoints_data.shape[0]):
                landmarks = keypoints_data[person_idx] / [w, h]
                landmarks_3d = np.hstack([landmarks, np.zeros((17, 1))])
                
                all_poses.append(landmarks_3d.astype(np.float32))
                all_visibilities.append(conf_data[person_idx].astype(np.float32))

            data.all_poses = all_poses
            data.all_visibilities = all_visibilities
            data.landmarks = all_poses[0]
            data.visibility = all_visibilities[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                box = result.boxes[0].xyxy[0].cpu().numpy()
                data.bounding_box = tuple(map(int, box))
        
        return data

    def _draw_landmarks(self, frame: np.ndarray, tracking_data: TrackingData) -> np.ndarray:
        if not tracking_data.has_person or not tracking_data.all_poses:
            return frame

        h, w = frame.shape[:2]
        colors = [(255, 100, 0), (0, 255, 255), (255, 255, 0), (0, 165, 255)]
        
        # Increase this to 0.6 if limbs are still "hallucinating" or flickering
        MIN_VISIBILITY = 0.55 

        for p_idx, landmarks in enumerate(tracking_data.all_poses):
            color = colors[p_idx % len(colors)]
            vis = tracking_data.all_visibilities[p_idx]
            
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if vis[start_idx] < MIN_VISIBILITY or vis[end_idx] < MIN_VISIBILITY:
                    continue

                start_pt = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                end_pt = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                
                # Filter points pinned to the bottom (likely missed detections)
                if start_pt[1] >= h - 5 or end_pt[1] >= h - 5:
                    continue
                    
                cv2.line(frame, start_pt, end_pt, color, 2, cv2.LINE_AA)

            for i, kp in enumerate(landmarks):
                if vis[i] > MIN_VISIBILITY:
                    pt = (int(kp[0] * w), int(kp[1] * h))
                    if pt[1] < h - 5:
                        cv2.circle(frame, pt, 4, (0, 255, 0), -1)

        return frame

    def close(self):
        self.model = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()