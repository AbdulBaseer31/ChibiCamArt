"""
main.py - Orchestrator for ChibiCam.
"""
import argparse
import sys
import time
import numpy as np
import torch
import cv2
import asyncio
import uvicorn
import threading

# Import trackers
from tracker import PoseTracker
from old_tracker import PoseTracker as MediaPipeTracker
from camera import CameraStream
from model_engine import ArtGenerator
from display import ScreenManager, ViewMode
from filters import FilterEngine
from server import app, manager
from pookie import RibbonEffect # <--- IMPORTED HERE

def parse_arguments():
    parser = argparse.ArgumentParser(description="ChibiCam")
    parser.add_argument("--camera", "-c", type=int, default=None)
    parser.add_argument("--view-mode", "-v", choices=["webcam_only", "wireframe_only", "matrix_only", "glitch_only", "terminal_only", "hologram_only", "dot_field_only"], default="webcam_only")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--fullscreen", "-f", action="store_true")
    parser.add_argument("--webview", "-w", action="store_true")
    return parser.parse_args()

def select_camera():
    available_cameras = []
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    if not available_cameras: return None
    print(f"\nAvailable Cameras: {available_cameras}")
    choice = input(f"Select camera index {available_cameras}: ")
    return int(choice) if choice.isdigit() else available_cameras[0]

async def async_main(args=None):
    if args is None: args = parse_arguments()
    
    # 1. Selection Menus
    print("\n[1] MediaPipe (Detailed) | [2] YOLOv8m (Fast)")
    tracker_type = "mediapipe" if input("Select tracker [1]: ") != "2" else "yolo"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.camera is None: args.camera = select_camera()
    
    # 2. Init Hardware/Modules
    target_res = (640, 480)
    camera = CameraStream(source=args.camera, target_resolution=target_res)
    camera.start()
    
    if tracker_type == "mediapipe":
        tracker = MediaPipeTracker(model_complexity=1, enable_segmentation=True)
    else:
        tracker = PoseTracker(confidence=0.5, device=device)
    
    generator = None
    filter_engine = FilterEngine()
    ribbon = RibbonEffect() # <--- INITIALIZED HERE
    
    display = ScreenManager(window_name="ChibiCam", show_debug=args.debug, fullscreen=args.fullscreen)
    display.set_model_name("MediaPipe" if tracker_type == "mediapipe" else "YOLOv8m")
    
    # Map view mode
    mode_map = {
        "webcam_only": ViewMode.WEBCAM_ONLY,
        "wireframe_only": ViewMode.WIREFRAME_ONLY,
        "matrix_only": ViewMode.MATRIX_ONLY,
        "glitch_only": ViewMode.GLITCH_ONLY,
        "terminal_only": ViewMode.TERMINAL_ONLY,
        "hologram_only": ViewMode.HOLOGRAM_ONLY,
        "dot_field_only": ViewMode.DOT_FIELD_ONLY
    }
    display.set_view_mode(mode_map[args.view_mode])

    # 3. Start Server
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error"), daemon=True).start()

    try:
        while True:
            t_start = time.time()
            frame = camera.get_latest_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # Settings from Frontend
            is_wireframe = (display.view_mode == ViewMode.WIREFRAME_ONLY) or (manager.settings.get("view_mode") == "wireframe")
            is_matrix = (display.view_mode == ViewMode.MATRIX_ONLY) or (manager.settings.get("view_mode") == "matrix")
            is_glitch = (display.view_mode == ViewMode.GLITCH_ONLY) or (manager.settings.get("view_mode") == "glitch")
            is_terminal = (display.view_mode == ViewMode.TERMINAL_ONLY) or (manager.settings.get("view_mode") == "terminal")
            is_hologram = (display.view_mode == ViewMode.HOLOGRAM_ONLY) or (manager.settings.get("view_mode") == "hologram")
            is_dot_field = (display.view_mode == ViewMode.DOT_FIELD_ONLY) or (manager.settings.get("view_mode") == "dot_field")
            needs_vis = is_wireframe or is_matrix or is_glitch or is_terminal or is_hologram or is_dot_field or display.show_debug or manager.settings.get("show_wireframe", False)
            
            # TRACKING
            tracking_data, tracking_vis = tracker.process_frame(
                frame, 
                return_visualization=needs_vis,
                black_background=is_wireframe
            )

            # STYLIZATION
            apply_stylize = manager.settings.get("apply_stylize", False)
            stylize_mode = manager.settings.get("stylize_mode", "face")
            
            stylized_frame = frame.copy()
            
            # Auto-apply Matrix effect when in Matrix view mode
            if is_matrix:
                stylized_frame = filter_engine.apply_matrix_filter(frame, tracking_data)
            elif is_glitch:
                stylized_frame = filter_engine.apply_glitch_filter(frame, tracking_data)
            elif is_terminal:
                stylized_frame = filter_engine.apply_terminal_filter(frame, tracking_data)
            elif is_hologram:
                stylized_frame = filter_engine.apply_hologram_filter(frame, tracking_data)
            elif is_dot_field:
                stylized_frame = filter_engine.apply_dot_field_filter(frame, tracking_data)
            elif apply_stylize:
                if "_cv" in stylize_mode:
                    first_person = None
                    if tracker_type == "mediapipe" and tracking_data.has_person:
                        class SimpleTracker: pass
                        first_person = SimpleTracker()
                        first_person.landmarks = tracking_data.pose_landmarks
                    elif isinstance(tracking_data, list) and len(tracking_data) > 0:
                        first_person = tracking_data[0]
                    
                    if stylize_mode == "anime_cv":
                        stylized_frame = filter_engine.apply_anime_filter(frame)
                    elif stylize_mode == "chibi_cv":
                        stylized_frame = filter_engine.apply_chibi_filter(frame, first_person)
                    elif stylize_mode == "matrix_cv":
                        stylized_frame = filter_engine.apply_matrix_filter(frame, tracking_data)
                else:
                    # PyTorch Neural Models
                    if generator is None or getattr(generator, 'model_type', None) != stylize_mode:
                        if generator: generator.close()
                        generator = ArtGenerator(model_path=None, input_size=(512,512), device=device, model_type=stylize_mode)
                    stylized_frame = generator.generate_with_compositing(frame, tracking_data)
            elif is_wireframe:
                stylized_frame = tracking_vis if tracking_vis is not None else np.zeros_like(frame)

            # GLOBAL OVERLAYS (Must happen BEFORE display.render)
            # This ensures the ribbon is always visible on top of the Arch Linux/Matrix void
            stylized_frame = ribbon.draw(stylized_frame, tracking_data)

            frame = ribbon.draw(frame, tracking_data)

            # Metrics and Networking
            fps = int(1.0 / (time.time() - t_start)) if (time.time() - t_start) > 0 else 0
            entities = 0
            if tracker_type == "mediapipe":
                entities = 1 if tracking_data.has_person else 0
            else:
                entities = len(tracking_data) if tracking_data else 0

            asyncio.create_task(manager.broadcast_frame(stylized_frame, {"fps": fps, "entities": entities}))

            # RENDER TO SCREEN (This captures the keypress via cv2.waitKey internally)
            if not display.render(stylized_frame=stylized_frame, original_frame=frame, tracking_overlay=tracking_vis):
                break
            
            # HANDLE INPUT EVENTS (Processed after render for the next frame)
            if hasattr(display, '_last_key_result') and display._last_key_result == 'toggle_ribbon':
                ribbon.toggle()
                display._last_key_result = None

            await asyncio.sleep(0.001)

    finally:
        camera.stop()
        display.close()
        cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    if args.webview:
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open('http://localhost:8000')).start()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()