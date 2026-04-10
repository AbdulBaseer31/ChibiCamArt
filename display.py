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
    WEBCAM_ONLY = auto()       # Original camera feed only
    WIREFRAME_ONLY = auto()    # Wireframe on black background
    MATRIX_ONLY = auto()       # Matrix binary rain effect
    GLITCH_ONLY = auto()       # Glitch digital artifact effect
    TERMINAL_ONLY = auto()     # Terminal effect with interactive buttons
    HOLOGRAM_ONLY = auto()     # Hologram effect
    DOT_FIELD_ONLY = auto()    # Dot field effect
    
    def next(self) -> "ViewMode":
        """Cycle to the next view mode."""
        modes = list(ViewMode)
        current_index = modes.index(self)
        return modes[(current_index + 1) % len(modes)]

class MenuState(Enum):
    """
    Menu states for keyboard navigation.
    """
    MAIN = auto()
    INTERACTIVE = auto()
    FILTERS = auto()
    PROPS = auto()
    CAMERA_SELECT = auto()


class ScreenManager:
    """
    Manages OpenCV display windows with text overlays disabled.
    """
    
    def __init__(
        self,
        window_name: str = "ChibiCam - Interactive Art Mirror",
        display_resolution: Optional[Tuple[int, int]] = None,
        show_fps: bool = True,    # Restored so main.py doesn't crash
        show_debug: bool = True,  # Restored so main.py doesn't crash
        fullscreen: bool = False,
        **kwargs                  # Catch-all for any other stray args
    ) -> None:
        self.window_name = window_name
        self.display_resolution = display_resolution
        self.fullscreen = fullscreen
        
        # We store these just in case other parts of your code check them, 
        # even though we aren't drawing them anymore.
        self.show_fps = show_fps
        self.show_debug = show_debug
        
        # Internal state
        self._window_created = False
        self._is_running = False
        self.menu_state = MenuState.MAIN
        self.view_mode: ViewMode = ViewMode.WIREFRAME_ONLY
        self.show_ui = False
        
        # Performance tracking (kept for logic, but not displayed)
        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0
        self._timing_data: Dict[str, float] = {}
        
        print(f"[ScreenManager] UI-Free Mode Initialized.")
    
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

    def _draw_help_popup(self, frame: np.ndarray) -> np.ndarray:
        if not self.show_help:
            return frame

        lines = []
        if self.view_mode == ViewMode.WIREFRAME_ONLY:
            lines = [
                "Available gestures:",
                "- Salt Drop (pinch)",
                "- Victory (V pose)",
                "- Thumbs up and Thumbs down",
                "- Explosion (Clap or two fists touching)"
            ]
        elif self.view_mode == ViewMode.HOLOGRAM_ONLY:
            lines = [
                "Instructions:",
                "- Keep palm open to Activate UI",
                "- Close fist to deactivate UI",
                "- Pinch on element to select it",
                "- Each element behaves differently with",
                "  lasers and bullets"
            ]
        else:
            self.show_help = False
            return frame

        h, w = frame.shape[:2]
        
        # Calculate optimal box size
        max_width = 0
        for line in lines:
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            if size[0] > max_width:
                max_width = size[0]
                
        box_width = max_width + 40
        box_height = len(lines) * 25 + 30
        
        x_start = (w - box_width) // 2
        y_start = (h - box_height) // 2
        
        # Draw background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + box_width, y_start + box_height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x_start, y_start), (x_start + box_width, y_start + box_height), (255, 200, 0), 2)
        
        # Render text lines
        y_text = y_start + 30
        for line in lines:
            cv2.putText(frame, line, (x_start + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y_text += 25
            
        return frame

    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Mode Tag
        mode_text = f"MODE: {self.view_mode.name.replace('_', ' ')}"
        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (w - text_size[0] - 20, 10), (w - 10, 35), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (w - text_size[0] - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Menu Instructions
        if self.menu_state == MenuState.MAIN:
            instructions = [
                "=== MAIN MENU ===", 
                "1: Interactive Filters", 
                "2: Filters", 
                "3: Props",
                "L: Select Camera",
                "C: Clear (Webcam)",
                "Q/ESC: Quit"
            ]
            if self.view_mode in [ViewMode.WIREFRAME_ONLY, ViewMode.HOLOGRAM_ONLY]:
                instructions.insert(1, "H: Toggle Help")

        elif self.menu_state == MenuState.INTERACTIVE:
            instructions = ["=== INTERACTIVE ===", "1: Wireframe", "2: Hologram", "0/ESC: Back"]
        
        elif self.menu_state == MenuState.FILTERS:
            instructions = ["=== FILTERS ===", "1: Matrix", "2: Glitch", "3: Terminal", "4: Dot Field", "0/ESC: Back"]
        
        elif self.menu_state == MenuState.PROPS:
            instructions = ["=== PROPS ===", "1: Pookie Mode (Ribbon)", "0/ESC: Back"]
            
        elif self.menu_state == MenuState.CAMERA_SELECT:
            instructions = ["=== CAMERAS ===", "1-0: Set Camera (Index 1-10)", "ESC: Back"]

        # Render instructions bottom-up
        y_start = h - 30
        for i, instruction in enumerate(reversed(instructions)):
            y_pos = y_start - (i * 20)
            cv2.putText(frame, instruction, (w - 240, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, instruction, (w - 240, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def update_timing(self, stage_name: str, duration_seconds: float) -> None:
        self._timing_data[stage_name] = duration_seconds
    
    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name
    
    def set_view_mode(self, mode: ViewMode) -> None:
        self.view_mode = mode
        # Auto-disable help if switching out of an interactive mode
        if mode not in [ViewMode.WIREFRAME_ONLY, ViewMode.HOLOGRAM_ONLY]:
            self.show_help = False
        print(f"[ScreenManager] View mode: {mode.name}")
    
    def cycle_view_mode(self) -> None:
        self.set_view_mode(self.view_mode.next())
    
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
        
        if getattr(self, 'show_ui', False):
            display_frame = self._draw_fps_overlay(display_frame)
            display_frame = self._draw_debug_overlay(display_frame)
            display_frame = self._draw_instructions(display_frame)
            display_frame = self._draw_help_popup(display_frame)
        
        cv2.imshow(self.window_name, display_frame)
        
        # ---------- Keyboard Input Handling ----------
        key = cv2.waitKey(1) & 0xFF
        self._last_key_result = None
        
        # Global Navigation / Exits
        if key == ord('q'):
            print("[ScreenManager] Quit signal received")
            return False
        elif key == 27: # ESC key
            if self.menu_state != MenuState.MAIN:
                self.menu_state = MenuState.MAIN
            else:
                return False
        elif key == ord('0'): 
            if self.menu_state == MenuState.CAMERA_SELECT:
                self._last_key_result = 'set_cam_9'
                self.menu_state = MenuState.MAIN
            else:
                self.menu_state = MenuState.MAIN

        # Menu Routing
        elif self.menu_state == MenuState.MAIN:
            if key == ord('1'): self.menu_state = MenuState.INTERACTIVE
            elif key == ord('2'): self.menu_state = MenuState.FILTERS
            elif key == ord('3'): self.menu_state = MenuState.PROPS
            elif key in [ord('l'), ord('L')]: self.menu_state = MenuState.CAMERA_SELECT
            elif key == ord('c'): self.set_view_mode(ViewMode.WEBCAM_ONLY)
                
        elif self.menu_state == MenuState.INTERACTIVE:
            if key == ord('1'): 
                self.set_view_mode(ViewMode.WIREFRAME_ONLY)
                self.menu_state = MenuState.MAIN
            elif key == ord('2'): 
                self.set_view_mode(ViewMode.HOLOGRAM_ONLY)
                self.menu_state = MenuState.MAIN
                
        elif self.menu_state == MenuState.FILTERS:
            if key == ord('1'): 
                self.set_view_mode(ViewMode.MATRIX_ONLY)
                self.menu_state = MenuState.MAIN
            elif key == ord('2'): 
                self.set_view_mode(ViewMode.GLITCH_ONLY)
                self.menu_state = MenuState.MAIN
            elif key == ord('3'): 
                self.set_view_mode(ViewMode.TERMINAL_ONLY)
                self.menu_state = MenuState.MAIN
            elif key == ord('4'): 
                self.set_view_mode(ViewMode.DOT_FIELD_ONLY)
                self.menu_state = MenuState.MAIN
                
        elif self.menu_state == MenuState.PROPS:
            if key == ord('1'):
                self._last_key_result = 'toggle_pookie'
                self.menu_state = MenuState.MAIN
                
        elif self.menu_state == MenuState.CAMERA_SELECT:
            if ord('1') <= key <= ord('9'):
                cam_idx = key - ord('1')
                self._last_key_result = f'set_cam_{cam_idx}'
                self.menu_state = MenuState.MAIN

        # Global Utilities
        if key == ord('f'): self._toggle_fullscreen()
        elif key == ord('d'): self.show_debug = not self.show_debug
        elif key == ord('u'): self.show_ui = not getattr(self, 'show_ui', False)
        elif key == ord(' '): 
            print("[ScreenManager] Paused - press any key to continue")
            cv2.waitKey(0)
        elif key == ord('h'):
            if self.view_mode in [ViewMode.WIREFRAME_ONLY, ViewMode.HOLOGRAM_ONLY]:
                self.show_help = not self.show_help
        
        return True
    
    def _create_split_view(self, original: np.ndarray, stylized: np.ndarray, tracking_overlay: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = original.shape[:2]
        if stylized.shape[:2] != (h, w):
            stylized = cv2.resize(stylized, (w, h))
        
        left_frame = original.copy()
        cv2.putText(left_frame, "Original", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(stylized, "Chibi Style", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return np.hstack([left_frame, stylized])
    
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