import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import argparse
import time

try:
    from diffusers import AutoPipelineForImage2Image, LCMScheduler
except ImportError:
    print("Please install diffusers: pip install diffusers transformers accelerate")
    import sys
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dataset for Chibi LoRA Distillation")
    parser.add_argument("--lora-path", type=str, default="../style05.safetensors", help="Path to the Chibi LoRA file")
    parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base SD model")
    parser.add_argument("--out-dir", type=str, default="../data/chibi_dataset", help="Output directory for dataset")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--max-images", type=int, default=1000, help="Number of image pairs to generate")
    parser.add_argument("--use-lcm", action="store_true", default=True, help="Use Latent Consistency Models for speed")
    parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame to ensure dataset variance")
    return parser.parse_args()

class DistillationGenerator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading YOLOv8 for face/person tracking...")
        # yolov8n-face.pt is better for just faces, but yolov8n.pt (class 0=person) works if we crop the top half.
        self.yolo = YOLO('yolov8n.pt') 
        
        print(f"Loading Diffusion Pipeline on {self.device}...")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.args.base_model, 
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        if self.args.use_lcm:
            print("Applying LCM Scheduler for ultra-fast generation...")
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
        
        print(f"Loading User Chibi LoRA from {self.args.lora_path}...")
        try:
            self.pipe.load_lora_weights(".", weight_name=self.args.lora_path, adapter_name="chibi")
        except Exception as e:
            print(f"Failed to load LoRA: {e}. Check the path.")
            sys.exit(1)
            
        if self.args.use_lcm:
            self.pipe.set_adapters(["lcm", "chibi"], adapter_weights=[1.0, 0.85])
        
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # Setup dataset directories
        self.src_dir = os.path.join(self.args.out_dir, "original")
        self.dst_dir = os.path.join(self.args.out_dir, "chibi")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dst_dir, exist_ok=True)
        
        # Check existing images to resume counting
        existing = len([f for f in os.listdir(self.src_dir) if f.endswith('.jpg')])
        self.pair_count = existing
        print(f"Found {existing} existing images in dataset.")

    def get_face_crop(self, frame: np.ndarray, box: list) -> np.ndarray:
        """Extracts a square crop focusing on the head/face area of a person bounding box."""
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
        
        # Assume face is in the top 40% of the bounding box
        face_cy = y1 + int(h * 0.2)
        face_cx = x1 + (w // 2)
        
        # Make a square crop. Width of face is usually ~50% of body width
        size = int(max(w, h) * 0.6)
        
        crop_x1 = max(0, face_cx - size // 2)
        crop_x2 = min(frame.shape[1], face_cx + size // 2)
        crop_y1 = max(0, face_cy - size // 2)
        crop_y2 = min(frame.shape[0], face_cy + size // 2)
        
        # Force exact square if cutoff by edge
        actual_w = crop_x2 - crop_x1
        actual_h = crop_y2 - crop_y1
        if actual_w != actual_h:
            min_dim = min(actual_w, actual_h)
            crop_x2 = crop_x1 + min_dim
            crop_y2 = crop_y1 + min_dim
            
        if crop_x2 - crop_x1 < 64:
            return None
            
        return frame[crop_y1:crop_y2, crop_x1:crop_x2]

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
        print("Starting dataset generation. Press 'q' to stop.")
        
        frame_idx = 0
        
        while cap.isOpened() and self.pair_count < self.args.max_images:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % self.args.frame_skip != 0:
                cv2.imshow("Dataset Generator", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
                
            # 1. Detect humans
            results = self.yolo(frame, classes=[0], verbose=False)
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.6: continue
                        
                    # 2. Extract squared face crop
                    crop_img = self.get_face_crop(frame, box.xyxy[0])
                    if crop_img is None: continue
                        
                    # 3. Resize to standard training size (e.g. 512x512 for SD)
                    crop_resized = cv2.resize(crop_img, (512, 512))
                    pil_crop = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                    
                    # 4. Generate Chibi using Diffusers
                    steps = 4 if self.args.use_lcm else 20
                    guidance = 1.2 if self.args.use_lcm else 7.0
                    
                    t0 = time.time()
                    chibi_pil = self.pipe(
                        prompt="masterpiece, best quality, chibi figure, cute anime style, highly detailed 8k face",
                        negative_prompt="ugly, bad anatomy, deformed, realistic, text, watermark",
                        image=pil_crop,
                        strength=0.55, # 0.55 heavily stylized but retains identity shape
                        num_inference_steps=steps,
                        guidance_scale=guidance
                    ).images[0]
                    t1 = time.time()
                    
                    # 5. Save Pair
                    filename = f"pair_{self.pair_count:05d}.jpg"
                    
                    # Save at 128x128 or 256x256 for the lightweight CPU model training later to save disk space
                    save_size = (128, 128)
                    
                    cv2.imwrite(
                        os.path.join(self.src_dir, filename), 
                        cv2.resize(crop_resized, save_size)
                    )
                    
                    chibi_cv2 = cv2.cvtColor(np.array(chibi_pil), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(self.dst_dir, filename), 
                        cv2.resize(chibi_cv2, save_size)
                    )
                    
                    self.pair_count += 1
                    print(f"[{self.pair_count}/{self.args.max_images}] Generated pair (Inference time: {t1-t0:.2f}s)")
                    
                    # Visualize
                    combined = np.hstack((cv2.resize(crop_resized, (256, 256)), cv2.resize(chibi_cv2, (256, 256))))
                    cv2.imshow("Original -> Chibi", combined)
                    cv2.waitKey(1)
                    
                    if self.pair_count >= self.args.max_images:
                        break
                        
            cv2.imshow("Dataset Generator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDataset generation completed. Saved {self.pair_count} pairs to {self.args.out_dir}")

if __name__ == "__main__":
    args = parse_args()
    gen = DistillationGenerator(args)
    gen.run()
