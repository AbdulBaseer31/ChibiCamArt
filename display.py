"""
display.py

High-performance display rendering module with FPS monitoring.
Handles OpenCV window management and performance overlay rendering.
"""

import time
from enum import Enum, auto
from typing import Optional, Tuple, Dict, List
from collections import deque
import numpy as np
import cv2


class ViewMode(Enum):
    """
    Display view modes for the application.
    """
    WEBCAM_ONLY = auto()       # 1: Original camera feed only
    WIREFRAME_ONLY = auto()    # 2: Wireframe on black background
    MATRIX_ONLY = auto()       # 3: Matrix binary rain effect
    GLITCH_ONLY = auto()       # 4: Glitch digital artifact effect
    TERMINAL_ONLY = auto()     # 5: Terminal effect with interactive buttons
    HOLOGRAM_ONLY = auto()     # 6: Hologram effect
    DOT_FIELD_ONLY = auto()    # 7: Dot field effect
    
    def next(self) -> "ViewMode":
        """Cycle to the next view mode."""
        modes = list(ViewMode)
        current_index = modes.index(self)
        return modes[(current_index + 1) % len(modes)]


class ScreenManager:
    """
    Manages OpenCV display windows with performance monitoring.
    
    Provides real-time FPS counter, frame timing visualization,
    and clean window lifecycle management.
    """
    
    def __init__(
        self,
        window_name: str = "ChibiCam - Interactive Art Mirror",
        display_resolution: Optional[Tuple[int, int]] = None,
        show_fps: bool = True,
        show_debug: bool = True,
        fullscreen: bool = False
    ) -> None:
        self.window_name = window_name
        self.display_resolution = display_resolution
        self.show_fps = show_fps
        self.show_debug = show_debug
        self.fullscreen = fullscreen
        self._last_key_result = None
        
        # Window state
        self._window_created = False
        self._is_running = False
        
        # Performance tracking
        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0
        
        # Pipeline timing tracking
        self._timing_data: Dict[str, float] = {}
        
        # UI state
        self._show_original = False  # Legacy toggle for side-by-side
        self.view_mode: ViewMode = ViewMode.WIREFRAME_ONLY  # Default: single output
        self._current_fps_color = (0, 255, 0)  # Green (good)
        self.model_name: str = "unknown"  # Track which pose model is being used
        self.device: str = "unknown"      # Track which compute device is being used
        
        print(f"[ScreenManager] Initialized: '{window_name}'")
    
    def create_window(self) -> None:
        if self._window_created:
            return
        
        # Create window
        if self.fullscreen:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            
            # Set initial size if resolution specified
            if self.display_resolution:
                width, height = self.display_resolution
                cv2.resizeWindow(self.window_name, width, height)
        
        # Move window to center-ish of screen
        cv2.moveWindow(self.window_name, 100, 100)
        
        self._window_created = True
        self._is_running = True
        print(f"[ScreenManager] Window created ({'fullscreen' if self.fullscreen else 'windowed'})")
    
    def _calculate_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        return 0.0
    
    def _update_performance_metrics(self) -> None:
        current_time = time.perf_counter()
        frame_time = current_time - self._last_frame_time
        self._last_frame_time = current_time
        
        self._frame_times.append(frame_time)
        self._fps = self._calculate_fps()
        
        # Update FPS color based on performance
        if self._fps >= 20:
            self._current_fps_color = (0, 255, 0)  # Green: excellent
        elif self._fps >= 15:
            self._current_fps_color = (0, 255, 255)  # Yellow: acceptable
        elif self._fps >= 10:
            self._current_fps_color = (0, 165, 255)  # Orange: warning
        else:
            self._current_fps_color = (0, 0, 255)  # Red: poor
    
    def _draw_fps_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.show_fps:
            return frame
        
        fps_text = f"FPS: {self._fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        cv2.rectangle(frame, (10, 10), (15 + text_size[0], 35), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self._current_fps_color, 2, cv2.LINE_AA)
        return frame
    
    def _draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.show_debug:
            return frame
        
        y_offset = 60
        line_height = 25
        debug_lines: List[str] = []
        
        debug_lines.append(f"Model: {self.model_name}")
        debug_lines.append(f"Device: {self.device.upper()}")
        
        if self._frame_times:
            avg_ms = (sum(self._frame_times) / len(self._frame_times)) * 1000
            debug_lines.append(f"Frame: {avg_ms:.1f}ms")
        
        total_pipeline = 0.0
        for stage, timing in self._timing_data.items():
            debug_lines.append(f"{stage}: {timing*1000:.1f}ms")
            total_pipeline += timing
        
        if total_pipeline > 0:
            debug_lines.append(f"Total: {total_pipeline*1000:.1f}ms")
        
        h, w = frame.shape[:2]
        debug_lines.append(f"Res: {w}x{h}")
        
        for i, line in enumerate(debug_lines):
            y_pos = y_offset + (i * line_height)
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (10, y_pos - 15), (15 + text_size[0], y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def update_timing(self, stage_name: str, duration_seconds: float) -> None:
        self._timing_data[stage_name] = duration_seconds
    
    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name
    
    def set_device(self, device: str) -> None:
        self.device = device
    
    def set_view_mode(self, mode: ViewMode) -> None:
        self.view_mode = mode
        print(f"[ScreenManager] View mode: {mode.name}")
    
    def cycle_view_mode(self) -> None:
        self.view_mode = self.view_mode.next()
        print(f"[ScreenManager] View mode: {self.view_mode.name}")
    
    def toggle_view(self) -> None:
        self.cycle_view_mode()
    
    def render(self, stylized_frame: np.ndarray, original_frame: Optional[np.ndarray] = None, tracking_overlay: Optional[np.ndarray] = None) -> bool:
        if not self._window_created:
            self.create_window()
        
        self._update_performance_metrics()
        
        # Determine base frame
        if self.view_mode == ViewMode.WEBCAM_ONLY:
            display_frame = original_frame.copy() if original_frame is not None else stylized_frame.copy()
        elif self.view_mode == ViewMode.WIREFRAME_ONLY:
            display_frame = tracking_overlay.copy() if tracking_overlay is not None else stylized_frame.copy()
        else: 
            display_frame = stylized_frame.copy()
        
        if self.display_resolution:
            target_w, target_h = self.display_resolution
            display_frame = cv2.resize(display_frame, (target_w, target_h))
        
        display_frame = self._draw_fps_overlay(display_frame)
        display_frame = self._draw_debug_overlay(display_frame)
        display_frame = self._draw_instructions(display_frame)
        
        cv2.imshow(self.window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        self._last_key_result = None
        
        if key == ord('q') or key == 27:
            print("[ScreenManager] Quit signal received")
            return False
        elif key == ord('1'): self.set_view_mode(ViewMode.WEBCAM_ONLY)
        elif key == ord('2'): self.set_view_mode(ViewMode.WIREFRAME_ONLY)
        elif key == ord('3'): self.set_view_mode(ViewMode.MATRIX_ONLY)
        elif key == ord('4'): self.set_view_mode(ViewMode.GLITCH_ONLY)
        elif key == ord('5'): self.set_view_mode(ViewMode.TERMINAL_ONLY)
        elif key == ord('6'): self.set_view_mode(ViewMode.HOLOGRAM_ONLY)
        elif key == ord('7'): self.set_view_mode(ViewMode.DOT_FIELD_ONLY)
        elif key == ord('t'): self.cycle_view_mode()
        elif key == ord('f'): self._toggle_fullscreen()
        elif key == ord('d'): self.show_debug = not self.show_debug
        elif key == ord('g'): 
            self._last_key_result = 'g' # Passes the G keypress back up to main.py
        elif key == ord(' '): 
            print("[ScreenManager] Paused - press any key to continue")
            cv2.waitKey(0)
        
        return True
    
    def _create_split_view(self, original: np.ndarray, stylized: np.ndarray, tracking_overlay: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = original.shape[:2]
        if stylized.shape[:2] != (h, w):
            stylized = cv2.resize(stylized, (w, h))
        
        left_frame = original.copy()
        cv2.putText(left_frame, "Original", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(stylized, "Chibi Style", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return np.hstack([left_frame, stylized])
    
    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        mode_text = f"MODE: {self.view_mode.name.replace('_', ' ')}"
        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (w - text_size[0] - 20, 10), (w - 10, 35), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (w - text_size[0] - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        instructions = [
            "Q/ESC: Quit", "1: Webcam", "2: Wireframe", "3: Matrix", 
            "4: Glitch", "5: Terminal", "6: Hologram", "7: Dot Field",
            "T: Cycle modes", "F: Fullscreen", 
            "D: Debug info", "G: Toggle Ribbon", "SPACE: Pause"
        ]
        
        y_start = h - 30
        for i, instruction in enumerate(reversed(instructions)):
            y_pos = y_start - (i * 20)
            cv2.putText(frame, instruction, (w - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, instruction, (w - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def _toggle_fullscreen(self) -> None:
        if not self._window_created: return
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    def get_fps(self) -> float:
        return self._fps
    
    def is_running(self) -> bool:
        return self._is_running
    
    def close(self) -> None:
        self._is_running = False
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
    
    def __enter__(self) -> "ScreenManager":
        self.create_window()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()