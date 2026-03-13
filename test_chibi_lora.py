import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
try:
    from diffusers import AutoPipelineForImage2Image, LCMScheduler
    from diffusers.utils import load_image
except ImportError:
    print("Please install diffusers: pip install diffusers transformers accelerate")
    import sys
    sys.exit(1)

# Configuration
LORA_PATH = "style05.safetensors"
# Assuming SD 1.5 base for most standard LoRAs, change to SDXL if needed
BASE_MODEL = "runwayml/stable-diffusion-v1-5" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LCM = True # Latent Consistency Models massively speed up CPU inference (from 50 steps to 4 steps)

class LightweightChibiPipeline:
    def __init__(self):
        print("Loading YOLOv8 for human tracking...")
        self.yolo = YOLO('yolov8n.pt') 
        
        print(f"Loading Diffusion Pipeline on {DEVICE}...")
        # To make it lightweight, we use torch.float16 if on GPU, float32 on CPU
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            BASE_MODEL, 
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        if USE_LCM:
            print("Applying LCM Scheduler for ultra-fast generation...")
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
        
        print(f"Loading User Chibi LoRA from {LORA_PATH}...")
        self.pipe.load_lora_weights(".", weight_name=LORA_PATH, adapter_name="chibi")
        
        if USE_LCM:
            self.pipe.set_adapters(["lcm", "chibi"], adapter_weights=[1.0, 0.8])
        
        self.pipe.to(DEVICE)
        
        # Optional: heavily reduces RAM/VRAM but slightly slower
        # self.pipe.enable_attention_slicing() 

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # 1. Detect humans
        results = self.yolo(frame, classes=[0], verbose=False) # class 0 is person
        result_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < 0.5:
                    continue
                    
                # Calculate bounding box dimensions
                w = x2 - x1
                h = y2 - y1
                
                # Expand box slightly to capture whole body/context
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                size = max(w + pad_x*2, h + pad_y*2)
                
                # Make square for SD aspect ratio
                crop_x1 = max(0, cx - size // 2)
                crop_x2 = min(frame.shape[1], cx + size // 2)
                crop_y1 = max(0, cy - size // 2)
                crop_y2 = min(frame.shape[0], cy + size // 2)
                
                # Crop and stylize
                crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop_img.shape[0] < 64 or crop_img.shape[1] < 64:
                    continue
                    
                # Convert to PIL
                pil_crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                
                # Generate using Diffusers (Img2Img)
                # Lower strength = closer to original image, faster. Higher = more Chibi style.
                steps = 4 if USE_LCM else 20
                guidance = 1.0 if USE_LCM else 7.5
                
                print(f"Generating Chibi for crop {pil_crop.size}...")
                chibi_pil = self.pipe(
                    prompt="masterpiece, best quality, chibi figure, cute, highly detailed",
                    negative_prompt="ugly, bad anatomy, deformed, realistic",
                    image=pil_crop,
                    strength=0.6,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                ).images[0]
                
                # Resize back
                chibi_cv2 = cv2.cvtColor(np.array(chibi_pil), cv2.COLOR_RGB2BGR)
                chibi_cv2 = cv2.resize(chibi_cv2, (crop_x2 - crop_x1, crop_y2 - crop_y1))
                
                # Paste back into frame (simple overwrite, ideally we'd use a soft mask or segmentation)
                result_frame[crop_y1:crop_y2, crop_x1:crop_x2] = chibi_cv2
                
                # Draw box
                cv2.rectangle(result_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
                cv2.putText(result_frame, "Chibi", (crop_x1, crop_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_frame

def main():
    print("Initializing Lightweight Chibi Pipeline...")
    pipeline = LightweightChibiPipeline()
    
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        styled_frame = pipeline.process_frame(frame)
        
        cv2.imshow("ChibiCam (LoRA + YOLOv8)", styled_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
