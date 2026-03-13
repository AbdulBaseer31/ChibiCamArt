import cv2
import numpy as np
import random
from contextlib import contextmanager

# matrix effect uses its own tracker and implements a stylized green binary rain
from matrix_effect import MatrixEffect
from glitch_effect import GlitchEffect
from terminalizer import TerminalEffect
from pookie import RibbonEffect
from hologram_effect import HologramEffect
from dot_field import DotGridEffect

class FilterEngine:
    """
    Lightweight CPU-bound OpenCV Image Filtering Engine.
    Simulates artistic styles like Anime, Ghibli, and Caricature/Chibi
    using classic computer vision techniques instead of heavy neural networks.
    """
    def __init__(self):
        # We can cache any persistent resources here
        # Initialize matrix effect helper so it doesn't re-create tracker each call
        self.matrix_effect = MatrixEffect()
        self.glitch_effect = GlitchEffect()
        self.terminal_effect = TerminalEffect()
        self.ribbon_effect = RibbonEffect()
        self.hologram_effect = HologramEffect()
        self.dot_field_effect = DotGridEffect()
        
    def _color_quantize(self, img, k=8):
        """Reduces the number of colors in an image (posterization)"""
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

    def apply_anime_filter(self, frame):
        """
        Fast Anime Filter using Bilateral Filtering to blur colors while
        preserving edges, and adaptive thresholding to draw ink lines.
        """
        # Downscale for performance
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))

        # 1. Edge detection using median blur and adaptive thresholding
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        
        # 2. Color smoothing using bilateral filter
        color = small.copy()
        for _ in range(2):
            color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
            
        # 3. Increase saturation drastically for the "anime" look
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:,:,1] = hsv[:,:,1] * 1.5 # Increase saturation by 50%
        hsv[:,:,2] = hsv[:,:,2] * 1.2 # Increase value (brightness)
        np.clip(hsv, 0, 255, out=hsv)
        hsv = np.array(hsv, dtype=np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 4. Combine color and edges
        cartoon_small = cv2.bitwise_and(color, color, mask=edges)
        
        # Upscale back to original size
        cartoon = cv2.resize(cartoon_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cartoon

    def apply_ghibli_filter(self, frame):
        """
        Studio Ghibli style filter.
        Washes out the shadows, boosts greens/blues, and applies a soft painterly effect,
        then sharpens it slightly so it isn't just a blur.
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        
        # Edge preserving filter creates a nice base painterly look
        painterly = cv2.edgePreservingFilter(small, flags=1, sigma_s=50, sigma_r=0.6)
        
        # Color shifting (boosting greens and blues)
        b, g, r = cv2.split(painterly)
        
        # Boost green subtly
        g = cv2.addWeighted(g, 1.2, np.zeros_like(g), 0, 10)
        # Boost blue subtly
        b = cv2.addWeighted(b, 1.2, np.zeros_like(b), 0, 15)
        
        shifted = cv2.merge((b, g, r))
        
        # Sharpen the image slightly to fight the blur
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(shifted, -1, kernel)
        
        # Soft bloom effect (overlay a blurred version of itself using screen blending)
        blur = cv2.GaussianBlur(sharpened, (0, 0), sigmaX=3, sigmaY=3)
        # screen blend formula: 1 - (1-a)*(1-b)
        shifted_f = sharpened.astype(np.float32) / 255.0
        blur_f = blur.astype(np.float32) / 255.0
        
        bloom = 1.0 - (1.0 - shifted_f) * (1.0 - blur_f)
        ghibli_small = (bloom * 255).astype(np.uint8)
        
        return cv2.resize(ghibli_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
    def apply_watercolor_filter(self, frame):
        """Fast watercolor simulation using OpenCV stylization. Boosts color drastically."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        res_small = cv2.stylization(small, sigma_s=60, sigma_r=0.6)
        
        hsv = cv2.cvtColor(res_small, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:,:,1] = hsv[:,:,1] * 1.5 # Increase saturation by 50%
        np.clip(hsv, 0, 255, out=hsv)
        hsv = np.array(hsv, dtype=np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)

    def apply_matrix_filter(self, frame, tracking_data=None):
        """
        Matrix-style filter: converts detected humans to falling green binary code
        using a separate MatrixEffect helper. If tracking_data is available it is
        passed along to avoid recomputing detection.
        """
        # delegate to MatrixEffect; tracking_data may be None
        return self.matrix_effect.process_frame(frame, tracking_data)
    
    def apply_glitch_filter(self, frame, tracking_data=None):
        """
        Glitch effect: creates digital artifacts, chromatic aberration, 
        and tracking errors when a person is detected.
        """
        # delegate to GlitchEffect; tracking_data may be None
        return self.glitch_effect.process_frame(frame, tracking_data)
    
    def apply_terminal_filter(self, frame, tracking_data=None):
        """
        Terminal effect: creates falling terminal commands with 
        interactive run/stop buttons and pitch black human silhouette.
        """
        # delegate to TerminalEffect; tracking_data may be None
        return self.terminal_effect.process_frame(frame, tracking_data)
    
    def toggle_ribbon(self):
        """Toggle the ribbon effect on/off"""
        self.ribbon_effect.toggle()
    
    def apply_ribbon_overlay(self, frame, tracking_data=None):
        """
        Apply ribbon overlay on top of any processed frame.
        This should be called after all other effects.
        """
        return self.ribbon_effect.draw(frame, tracking_data)
    
    def apply_hologram_filter(self, frame, tracking_data=None):
        """
        Hologram effect: creates glowing cyan holographic body
        with simplistic face and hand UI panels.
        """
        return self.hologram_effect.process_frame(frame, tracking_data)
    
    def apply_dot_field_filter(self, frame, tracking_data=None):
        """
        Dot field effect: creates 3D dot grid with displaced body
        and high-detail face/hand landmarks.
        """
        return self.dot_field_effect.process_frame(frame, tracking_data)

    def _distort_region(self, frame, center_x, center_y, radius, scale_factor):
        """
        Pulls pixels towards or pushes them away from a center point.
        """
        h, w = frame.shape[:2]
        
        # Calculate bounding box for the region to be distorted
        x_min = max(0, center_x - radius)
        x_max = min(w, center_x + radius)
        y_min = max(0, center_y - radius)
        y_max = min(h, center_y + radius)
        
        # If region is invalid
        if x_max <= x_min or y_max <= y_min:
            return frame
            
        # Output frame
        output = frame.copy()
        
        # Create a meshgrid just for the bounding box (massive performance boost)
        y, x = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Calculate distances to center
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx**2 + dy**2)
        
        # Create mapping (we only affect pixels within 'radius')
        mask = r < radius
        
        map_x = x.copy().astype(np.float32)
        map_y = y.copy().astype(np.float32)
        
        if scale_factor > 1.0:
            # Bulge effect
            power = 1.0 / scale_factor
            mapped_r = radius * (r / radius) ** power
            
            # Avoid division by zero
            r_safe = r.copy()
            r_safe[r_safe == 0] = 1 
            
            # Apply distortion to mask
            map_x[mask] = center_x + dx[mask] * (mapped_r[mask] / r_safe[mask])
            map_y[mask] = center_y + dy[mask] * (mapped_r[mask] / r_safe[mask])
        
        # Remap the region mapped by the coordinates
        distorted_roi = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Blend exactly within the circular mask
        roi_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        original_roi = frame[y_min:y_max, x_min:x_max]
        output[y_min:y_max, x_min:x_max] = np.where(roi_mask, distorted_roi, original_roi)
        
        return output

    def apply_chibi_filter(self, frame, tracking_data):
        """
        Caricature/Chibi filter.
        Uses tracking data to find the head and eyes.
        Enlarges the head slightly and enlarges the eyes drastically.
        Then applies a cartoon/anime color pass.
        """
        if tracking_data is None or not getattr(tracking_data, 'landmarks', None) is not None:
            # If no tracking data, fallback to anime filter
            return self.apply_anime_filter(frame)
            
        h, w = frame.shape[:2]
        landmarks = tracking_data.landmarks
        
        # Assuming YOLOv8 pose format where first few keypoints are head/face
        # KP 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
        # Note: MediaPipe full tracking has 33 points, but 0-10 are face. 0=nose, 2=R_eye, 5=L_eye (from user perspective)
        # We'll use a heuristic for both. We just need rough eyes and head center.
        
        if len(landmarks) >= 5:
            # Get bounding box of the face landmarks to estimate size
            face_pts = landmarks[:5]
            x_coords = face_pts[:, 0] * w
            y_coords = face_pts[:, 1] * h
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            face_w = x_max - x_min
            face_h = y_max - y_min
            
            # Heuristic to find eyes (assuming KP 1 and 2 are eyes)
            # Find the two points in the upper half of the face box that are furthest apart horizontally
            # If it's pure standard YOLO, 1 and 2 are eyes. MediaPipe 2 and 5 are eyes roughly.
            # Let's just find the center of the bounding box and bulge the top half.
            
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            
            # 1. Distort the head (make it larger, "bobblehead" effect)
            radius = int(max(face_w, face_h) * 1.5)
            if radius > 0:
                frame = self._distort_region(frame, center_x, center_y, radius, scale_factor=1.2)
                
            # 2. Enlarged eyes (bulge the eyes region)
            # Assuming eyes are slightly above the center
            eye_y = int(center_y - face_h * 0.1)
            eye_radius = int(face_w * 0.4)
            
            # Left eye (screen space right)
            eye_x_right = int(center_x + face_w * 0.25)
            if eye_radius > 0:
                frame = self._distort_region(frame, eye_x_right, eye_y, eye_radius, scale_factor=1.5)
                
            # Right eye (screen space left)
            eye_x_left = int(center_x - face_w * 0.25)
            if eye_radius > 0:
                frame = self._distort_region(frame, eye_x_left, eye_y, eye_radius, scale_factor=1.5)
                
        # Finish with a clean anime pass
        return self.apply_anime_filter(frame)