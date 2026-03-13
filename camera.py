"""
camera.py

High-performance camera capture module with background threading.
Ensures zero-blocking frame acquisition for real-time applications.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np
import cv2


class CameraStream:
    """
    A thread-safe camera capture class that grabs frames on a background thread.
    
    This design prevents buffer lag by always providing the most recent frame
    to the consumer, dropping frames if processing falls behind.
    
    Attributes:
        capture: cv2.VideoCapture instance
        source: Camera device index or video file path
        target_resolution: Desired output resolution as (width, height)
        fps_limit: Maximum frames to capture per second (0 for unlimited)
    """
    
    def __init__(
        self,
        source: int = 0,
        target_resolution: Optional[Tuple[int, int]] = None,
        fps_limit: int = 60
    ) -> None:
        """
        Initialize the camera stream.
        
        Args:
            source: Camera device index (default 0) or video file path
            target_resolution: Optional (width, height) to resize frames
            fps_limit: Target FPS for capture (camera may override)
        """
        self.source = source
        self.target_resolution = target_resolution
        self.fps_limit = fps_limit
        self._min_frame_time = 1.0 / fps_limit if fps_limit > 0 else 0
        
        # Threading components
        self._lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Frame buffer - always contains the latest frame
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = threading.Event()
        self._last_frame_time = 0.0
        
        # Initialize camera
        self.capture: Optional[cv2.VideoCapture] = None
        self._initialize_camera()
    
    def _initialize_camera(self) -> None:
        """
        Initialize the VideoCapture device with optimized settings.
        
        Raises:
            RuntimeError: If camera cannot be opened
        """
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Failed to open camera source: {self.source}. "
                "Check camera connection and device index."
            )
        
        # Set camera properties for low latency
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer depth
        self.capture.set(cv2.CAP_PROP_FPS, self.fps_limit)
        
        if self.target_resolution:
            width, height = self.target_resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify we can read a test frame
        ret, test_frame = self.capture.read()
        if not ret or test_frame is None:
            raise RuntimeError(
                "Camera opened but failed to capture test frame. "
                "Hardware may be in use by another application."
            )
        
        # Store initial frame
        self._latest_frame = test_frame
        self._frame_ready.set()
        
        print(f"[CameraStream] Initialized: {self.get_resolution()} @ {self.fps_limit} FPS target")
    
    def _capture_loop(self) -> None:
        """
        Background thread loop that continuously grabs frames.
        
        This loop runs independently of the main processing loop,
        ensuring the camera buffer never fills and causes lag.
        """
        while not self._stop_event.is_set():
            current_time = time.perf_counter()
            elapsed = current_time - self._last_frame_time
            
            # Rate limiting if needed
            if elapsed < self._min_frame_time:
                time.sleep(self._min_frame_time - elapsed)
                continue
            
            # Grab the latest frame (non-blocking)
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                
                if ret and frame is not None:
                    # Apply mirror flip (horizontal)
                    frame = cv2.flip(frame, 1)
                    
                    # Apply resolution scaling if needed
                    if self.target_resolution:
                        frame = cv2.resize(frame, self.target_resolution)
                    
                    # Thread-safe frame update (overwrites old frame)
                    with self._lock:
                        self._latest_frame = frame
                        self._last_frame_time = current_time
                    
                    self._frame_ready.set()
            else:
                time.sleep(0.001)  # Small sleep if camera unavailable
    
    def start(self) -> "CameraStream":
        """
        Start the background capture thread.
        
        Returns:
            Self for method chaining
        """
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return self
        
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True  # Thread dies with main program
        )
        self._capture_thread.start()
        print("[CameraStream] Background capture thread started")
        return self
    
    def get_latest_frame(self, wait: bool = False, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Retrieve the most recently captured frame.
        
        This method is thread-safe and returns immediately with the
        latest available frame. No frame queuing - always drops stale
        frames to maintain real-time sync.
        
        Args:
            wait: If True, block until a frame is available
            timeout: Maximum seconds to wait if wait=True
            
        Returns:
            NumPy array (H, W, 3) BGR image or None if unavailable
        """
        if wait:
            self._frame_ready.wait(timeout=timeout)
        
        with self._lock:
            if self._latest_frame is not None:
                # Return a copy to prevent modification of internal buffer
                return self._latest_frame.copy()
            return None
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current camera resolution.
        
        Returns:
            (width, height) tuple
        """
        if self.capture:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def is_running(self) -> bool:
        """
        Check if capture thread is active.
        
        Returns:
            True if background thread is running
        """
        return (
            self._capture_thread is not None 
            and self._capture_thread.is_alive()
            and not self._stop_event.is_set()
        )
    
    def stop(self) -> None:
        """
        Stop the capture thread and release camera resources.
        
        This should be called during shutdown to release hardware.
        """
        print("[CameraStream] Stopping capture...")
        self._stop_event.set()
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        self._frame_ready.clear()
        print("[CameraStream] Camera released")
    
    def __enter__(self) -> "CameraStream":
        """Context manager entry - starts capture."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops capture."""
        self.stop()
