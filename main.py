"""
main.py - Orchestrator for ArtCam.
"""
from cv2.gapi.core import cpu
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
from pookie import RibbonEffect 
from hologram_effect import HologramEffect 

def parse_arguments():
    parser = argparse.ArgumentParser(description="ArtCam")
    parser.add_argument("--camera", "-c", type=int, default=None)
    parser.add_argument("--view-mode", "-v", choices=["webcam_only", "wireframe_only", "matrix_only", "glitch_only", "terminal_only", "hologram_only", "dot_field_only"], default="webcam_only")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--fullscreen", "-f", action="store_true")
    parser.add_argument("--webview", "-w", action="store_true")
    parser.add_argument("--gpu", "-g", action="store_true", help="Force CUDA GPU acceleration (use when PyTorch CUDA is installed system-wide)")
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
    
    # GPU / CPU Selection
    cuda_available = torch.cuda.is_available()
    if args.gpu:
        device = "cuda"
        print(f"[ArtCam] GPU mode forced via --gpu flag")
        if cuda_available:
            print(f"[ArtCam] CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[ArtCam] WARNING: torch.cuda.is_available() returned False.")
            print(f"[ArtCam] Attempting CUDA anyway (system-wide PyTorch CUDA).")
    elif cuda_available:
        print(f"\n[ArtCam] GPU detected: {torch.cuda.get_device_name(0)}")
        print("[1] GPU (CUDA) | [2] CPU")
        gpu_choice = input("Select device [1]: ").strip()
        device = "cpu" if gpu_choice == "2" else "cuda"
    else:
        print("\n[ArtCam] No CUDA GPU detected. Running on CPU.")
        print("[ArtCam] TIP: Use --gpu flag if PyTorch CUDA is installed system-wide.")
        device = "cpu"
    
    print(f"[ArtCam] Using device: {device.upper()}")
    manager.settings["device"] = device
    
    if args.camera is None: args.camera = select_camera()
    
    # 2. Init Hardware/Modules
    target_res = (640, 480)
    camera = CameraStream(source=args.camera, target_resolution=target_res)
    camera.start()
    
    if tracker_type == "mediapipe":
        tracker = MediaPipeTracker(device=cpu)
    else:
        tracker = PoseTracker(confidence=0.5, device=device)
    
    generator = None
    filter_engine = FilterEngine()
    ribbon = RibbonEffect() 
    hologram_effect = HologramEffect() 
    
    # Stylization caching for frame skipping
    last_stylized_frame = None
    stylize_skip_count = 0
    STYLIZE_SKIP_RATE = 2  # Process every Nth frame if neural style is active
    
    display = ScreenManager(window_name="ArtCam", show_debug=args.debug, fullscreen=args.fullscreen)
    display.set_model_name("MediaPipe" if tracker_type == "mediapipe" else "YOLOv8m")
    display.set_device(device)
    
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
            is_matrix = (display.view_mode == ViewMode.MATRIX_ONLY)
            is_glitch = (display.view_mode == ViewMode.GLITCH_ONLY)
            is_terminal = (display.view_mode == ViewMode.TERMINAL_ONLY)
            is_hologram = (display.view_mode == ViewMode.HOLOGRAM_ONLY)
            is_dot_field = (display.view_mode == ViewMode.DOT_FIELD_ONLY)
            
            # Handle filter selection (only applies to webcam mode)
            current_filter = manager.settings.get("currentFilter", "none")
            is_glitch_filter = current_filter == "glitch"
            is_dot_field_filter = current_filter == "dot_field"
            is_matrix_filter = current_filter == "matrix"
            is_terminal_filter = current_filter == "terminal"
            
            # Handle pookie mode (applies to all modes)
            pookie_mode = manager.settings.get("pookieMode", False)
            ribbon.enabled = pookie_mode
            
            needs_vis = is_wireframe or is_matrix or is_glitch or is_terminal or is_hologram or is_dot_field or display.show_debug or manager.settings.get("show_wireframe", False)
            
            # TRACKING
            t0 = time.time()
            tracking_data, tracking_vis = tracker.process_frame(
                frame, 
                return_visualization=needs_vis,
                black_background=is_wireframe
            )
            display.update_timing("Tracking", time.time() - t0)

            # STYLIZATION
            apply_stylize = manager.settings.get("apply_stylize", False)
            stylize_mode = manager.settings.get("stylize_mode", "anime_cv")
            
            stylized_frame = frame.copy()
            
            # Apply filters when in webcam mode
            if display.view_mode == ViewMode.WEBCAM_ONLY:
                if is_glitch_filter:
                    stylized_frame = filter_engine.apply_glitch_filter(frame, tracking_data)
                elif is_dot_field_filter:
                    stylized_frame = filter_engine.apply_dot_field_filter(frame, tracking_data)
                elif is_matrix_filter:
                    stylized_frame = filter_engine.apply_matrix_filter(frame, tracking_data)
                elif is_terminal_filter:
                    stylized_frame = filter_engine.apply_terminal_filter(frame, tracking_data)
            
            # Apply view mode specific effects
            if is_matrix:
                stylized_frame = filter_engine.apply_matrix_filter(frame, tracking_data)
            elif is_glitch:
                stylized_frame = filter_engine.apply_glitch_filter(frame, tracking_data)
            elif is_terminal:
                stylized_frame = filter_engine.apply_terminal_filter(frame, tracking_data)
            elif is_hologram:
                stylized_frame = hologram_effect.process_frame(frame, tracking_data) 
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
                    
                    if stylize_mode == "chibi_cv":
                        stylized_frame = filter_engine.apply_chibi_filter(frame, first_person)
                    elif stylize_mode == "anime_cv":
                        stylized_frame = filter_engine.apply_anime_filter(frame, first_person)
                    elif stylize_mode == "ghibli_cv":
                        stylized_frame = filter_engine.apply_ghibli_filter(frame, first_person)
                    elif stylize_mode == "watercolor_cv":
                        stylized_frame = filter_engine.apply_watercolor_filter(frame, first_person)
                else:
                    # PyTorch Neural Models
                    if generator is None or getattr(generator, 'model_type', None) != stylize_mode:
                        if generator: generator.close()
                        generator = ArtGenerator(model_path=None, input_size=(512,512), device=device, model_type=stylize_mode)
                    
                    # Frame skipping logic for heavy neural styles
                    if last_stylized_frame is None or stylize_skip_count >= STYLIZE_SKIP_RATE:
                        stylized_frame = generator.generate_with_compositing(frame, tracking_data)
                        last_stylized_frame = stylized_frame.copy()
                        stylize_skip_count = 0
                    else:
                        stylize_skip_count += 1
                        stylized_frame = last_stylized_frame.copy() 
            
            # Handle wireframe overlay
            if is_wireframe:
                stylized_frame = tracking_vis if tracking_vis is not None else np.zeros_like(frame)
            
            # GLOBAL OVERLAYS (Must happen BEFORE display.render)
            if pookie_mode:
                if stylized_frame is not None:
                    stylized_frame = ribbon.draw(stylized_frame, tracking_data)
                if frame is not None:
                    frame = ribbon.draw(frame, tracking_data)

            # Metrics and Networking
            fps = int(1.0 / (time.time() - t_start)) if (time.time() - t_start) > 0 else 0
            entities = 0
            if tracker_type == "mediapipe":
                entities = 1 if tracking_data.has_person else 0
            else:
                entities = len(tracking_data) if tracking_data else 0

            asyncio.create_task(manager.broadcast_frame(stylized_frame, {"fps": fps, "entities": entities}))

            # RENDER TO SCREEN
            if not display.render(stylized_frame=stylized_frame, original_frame=frame, tracking_overlay=tracking_vis):
                break
            
            # HANDLE INPUT EVENTS (Processed from the nested menu state machine)
            if hasattr(display, '_last_key_result') and display._last_key_result:
                if display._last_key_result == 'toggle_pookie':
                    current_state = manager.settings.get("pookieMode", False)
                    manager.settings["pookieMode"] = not current_state
                    print(f"[Main] Toggled Ribbon via keyboard: {not current_state}")
                
                # Reset key state
                display._last_key_result = None

            await asyncio.sleep(0.001)

    finally:
        camera.stop()
        display.close()
        hologram_effect.close() 
        cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    if args.webview:
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open('http://localhost:8000')).start()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()