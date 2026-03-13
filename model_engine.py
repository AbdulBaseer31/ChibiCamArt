"""
model_engine.py

Core AI stylization engine using PyTorch with fp16 optimization.
Implements fast neural style transfer for Chibi anime art generation.
Optimized for RTX 4050 with 6GB VRAM constraints.
"""

import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Optional safetensors support
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


class ResidualBlock(nn.Module):
    """
    Residual block for the TransformerNet architecture.
    Maintains feature dimensions while learning residual mappings.
    """
    
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class TransformerNet(nn.Module):
    """
    Lightweight transformer network for fast neural style transfer.
    
    Architecture: Conv layers -> Residual blocks -> Upsampling
    Designed for real-time inference with minimal compute.
    
    Input: Batch of images (B, 3, H, W)
    Output: Stylized images (B, 3, H, W)
    """
    
    def __init__(self) -> None:
        super(TransformerNet, self).__init__()
        
        # Initial convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual blocks (lightweight: 5 blocks)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(5)
        ])
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        
        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolutions
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        
        # Residual blocks
        for block in self.res_blocks:
            y = block(y)
        
        # Upsampling
        y = self.relu(self.in4(self.upconv1(y)))
        y = self.relu(self.in5(self.upconv2(y)))
        
        # Final output (scaled to [0, 1])
        y = self.conv4(y)
        y = self.tanh(y)
        y = (y + 1) / 2  # Scale from [-1, 1] to [0, 1]
        
        return y


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for lightweight mobile inference."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightweightResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class LightweightFaceStyleNet(nn.Module):
    """Ultra-lightweight style transfer network for fast CPU inference on face crops."""
    def __init__(self) -> None:
        super().__init__()
        # Thinner initial layers (16 -> 32 -> 64 vs 32 -> 64 -> 128)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.in1 = nn.InstanceNorm2d(16, affine=True)
        
        self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(32, affine=True)
        
        self.conv3 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(64, affine=True)
        
        # Fewer, lighter residual blocks (4 blocks instead of 5)
        self.res_blocks = nn.ModuleList([
            LightweightResidualBlock(64) for _ in range(4)
        ])
        
        # Upsampling with ConvTranspose
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(32, affine=True)
        
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(16, affine=True)
        
        self.conv4 = nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        
        for block in self.res_blocks:
            y = block(y)
            
        y = self.relu(self.in4(self.upconv1(y)))
        y = self.relu(self.in5(self.upconv2(y)))
        
        y = self.conv4(y)
        y = self.tanh(y)
        return (y + 1) / 2


class ArtGenerator:
    """
    Real-time Chibi art style generator using Fast Neural Style Transfer.
    
    This class manages the PyTorch model on CUDA with fp16 precision
    to fit within 6GB VRAM while maintaining real-time performance.
    
    Attributes:
        device: CUDA device for computation
        model: TransformerNet style transfer model
        input_size: Target processing resolution
        use_fp16: Whether to use half-precision floating point
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (512, 512),
        use_fp16: bool = True,
        device: str = "cuda",
        model_type: str = "full"
    ) -> None:
        """
        Initialize the art generator with model loading.
        
        Args:
            model_path: Path to pretrained model weights (.pth, .safetensors, or .onnx)
            input_size: (width, height) for model input processing
            use_fp16: Enable half-precision to save VRAM (ignored for generic ONNX cpu)
            device: Device string ('cuda' or 'cpu')
            model_type: 'full' for full-frame TransformerNet, 'face' for LightweightFaceStyleNet
            
        Raises:
            RuntimeError: If CUDA unavailable and device='cuda'
        """
        self.input_size = input_size
        self.use_fp16 = use_fp16
        self.device = torch.device(device)
        self.model_type = model_type
        self.is_onnx = False
        self.ort_session = None

        if self.device.type == "cpu" and self.use_fp16:
            print("[ArtGenerator] WARNING: FP16 requested but device is CPU. Disabling FP16.")
            self.use_fp16 = False
        
        if device == "cuda" and not torch.cuda.is_available():
            print("[ArtGenerator] WARNING: CUDA requested but torch.cuda.is_available() is False.")
            print("[ArtGenerator] Proceeding anyway (system-wide PyTorch CUDA may work).")
        
        # Initialize model
        if self.model_type == "face":
            self.model = LightweightFaceStyleNet()
        else:
            self.model = TransformerNet()
            
        self.model.to(self.device)
        
        # Convert to half precision for VRAM savings (~50% reduction)
        if self.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            print("[ArtGenerator] Model converted to fp16")
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        else:
            print("[ArtGenerator] Warning: No pretrained weights loaded. Using random init.")
            print("[ArtGenerator] For Chibi style, download a pretrained FastNST model.")
        
        # Set model to evaluation mode (disables dropout/batch norm updates)
        self.model.eval()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size),
        ])
        
        # Post-processing for output
        self.to_pil = transforms.ToPILImage()
        
        # Pre-allocate tensors for efficiency
        self._dummy_input = torch.zeros(
            (1, 3, input_size[1], input_size[0]),
            device=self.device,
            dtype=torch.float16 if self.use_fp16 else torch.float32
        )
        
        # Warm up GPU with dummy inference
        self._warmup()
        
        # Memory management
        self._clear_cuda_cache()
        
        print(f"[ArtGenerator] Initialized on {self.device} @ {input_size}")
        self._print_memory_stats()
    
    def _load_weights(self, model_path: str) -> None:
        """
        Load pretrained model weights with support for .pth, .safetensors, and .onnx formats.
        
        Args:
            model_path: Path to weights file
        """
        try:
            file_ext = os.path.splitext(model_path)[1].lower()
            
            if file_ext == '.onnx':
                import onnxruntime as ort
                self.is_onnx = True
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == "cuda" else ['CPUExecutionProvider']
                
                # Optional OpenVINO acceleration if installed (Intel CPUs)
                if self.device.type == "cpu" and 'OpenVINOExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'OpenVINOExecutionProvider')
                    
                self.ort_session = ort.InferenceSession(model_path, providers=providers)
                print(f"[ArtGenerator] Loaded ONNX model from {model_path} with providers: {self.ort_session.get_providers()}")
                return
                
            if file_ext == '.safetensors':
                # Load safetensors format
                if not SAFETENSORS_AVAILABLE:
                    raise ImportError(
                        "safetensors package not installed. "
                        "Install with: pip install safetensors"
                    )
                state_dict = load_safetensors(model_path)
                print(f"[ArtGenerator] Loaded safetensors weights from {model_path}")
            else:
                # Load PyTorch pickle format (.pth, .pt)
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                
                print(f"[ArtGenerator] Loaded PyTorch weights from {model_path}")
            
            # Load weights into model
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[ArtGenerator] Weights loaded successfully")
            
        except Exception as e:
            print(f"[ArtGenerator] Error loading weights: {e}")
            raise
    
    def _warmup(self, num_iterations: int = 3) -> None:
        """
        Perform warmup iterations to initialize CUDA kernels.
        
        First CUDA operations have overhead; warmup eliminates
        this latency from the actual inference loop.
        
        Args:
            num_iterations: Number of warmup forward passes
        """
        print(f"[ArtGenerator] Warming up {self.device.type.upper()} ({num_iterations} iterations)...")
        
        if self.is_onnx:
            # Warmup ONNX
            import numpy as np
            ort_inputs = {self.ort_session.get_inputs()[0].name: self._dummy_input.cpu().numpy().astype(np.float32)}
            for _ in range(num_iterations):
                _ = self.ort_session.run(None, ort_inputs)
        else:
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.model(self._dummy_input)
        
        # Only synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # Wait for completion
        print("[ArtGenerator] Warmup complete")
    
    def _clear_cuda_cache(self) -> None:
        """
        Clear CUDA cache to free unused VRAM.
        Call periodically to prevent OOM errors during long runs.
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def _print_memory_stats(self) -> None:
        """
        Print current GPU memory usage statistics.
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            print(f"[ArtGenerator] VRAM: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
    
    def preprocess(
        self, 
        frame: np.ndarray,
        tracking_data: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Convert OpenCV BGR frame to model-ready tensor.
        
        Args:
            frame: NumPy array (H, W, 3) in BGR format
            tracking_data: Optional tracking data for region-of-interest
            
        Returns:
            Preprocessed tensor on target device
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for torchvision transforms
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Convert to fp16 if enabled
        if self.use_fp16:
            tensor = tensor.half()
        
        return tensor
    
    def postprocess(self, tensor: torch.Tensor, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Convert model output tensor back to OpenCV-compatible NumPy array.
        
        Args:
            tensor: Output tensor from model (1, 3, H, W)
            original_shape: (H, W) of original input for resizing
            
        Returns:
            NumPy array (H, W, 3) uint8 in BGR format
        """
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).cpu()
        
        # Convert fp16 back to fp32 for PIL
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        
        # Convert to PIL and resize to original
        pil_image = self.to_pil(tensor)
        pil_image = pil_image.resize((original_shape[1], original_shape[0]), Image.LANCZOS)
        
        # Convert back to NumPy and RGB to BGR
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        # Convert to uint8
        bgr_array = (bgr_array * 255).clip(0, 255).astype(np.uint8)
        
        return bgr_array
    
    def generate(
        self, 
        frame: np.ndarray,
        tracking_data: Optional[Any] = None
    ) -> np.ndarray:
        """
        Generate stylized Chibi art from input frame.
        
        This is the main inference method that:
        1. Preprocesses the input frame
        2. Runs model forward pass (no gradient computation)
        3. Postprocesses output back to OpenCV format
        
        Args:
            frame: Input BGR image from camera
            tracking_data: Optional pose data for pose-aware stylization
            
        Returns:
            Stylized BGR image as NumPy array
        """
        original_shape = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(frame, tracking_data)
        
        if self.is_onnx:
            # Inference via ONNX Runtime
            import numpy as np
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor.cpu().numpy().astype(np.float32)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            output_tensor = torch.from_numpy(ort_outs[0]).to(self.device)
        else:
            # Inference without gradient computation (critical for speed)
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
        
        # Postprocess
        stylized_frame = self.postprocess(output_tensor, original_shape)
        return stylized_frame
        
    def _get_face_bbox(self, frame: np.ndarray, tracking_data: Any) -> Optional[Tuple[int, int, int, int]]:
        """Extract a squared bounding box around the face from tracking landmarks."""
        if tracking_data is None or not tracking_data.has_person or tracking_data.landmarks is None:
            return None
        
        h, w = frame.shape[:2]
        # Common face landmark indices across YOLOv8/MediaPipe
        # MediaPipe uses 0-10 for face. YOLOv8 uses 0-4 for face.
        # We can just take the first 5 landmarks (Nose, Eyes, Ears)
        face_lms = tracking_data.landmarks[:5]
        
        # Some might not be visible (handled by visibility array if needed, but we'll just min/max here)
        x_coords = face_lms[:, 0] * w
        y_coords = face_lms[:, 1] * h
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add a large margin (ears to eyes is narrow; we want the whole head)
        face_w = x_max - x_min
        face_h = y_max - y_min
        
        # If the detection is bogus (e.g., side profile where eyes overlap)
        if face_w < 10 and face_h < 10:
            return None
            
        # Expanded padding to capture hair and chin
        pad_x = face_w * 1.5
        pad_y = face_h * 1.5
        
        x_min = max(0, int(x_min - pad_x))
        x_max = min(w, int(x_max + pad_x))
        y_min = max(0, int(y_min - pad_y * 1.2)) # More headroom
        y_max = min(h, int(y_max + pad_y * 0.8))
        
        # Make the box square for better style transfer
        box_w = x_max - x_min
        box_h = y_max - y_min
        size = max(box_w, box_h)
        
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        
        sq_x_min = max(0, cx - size // 2)
        sq_x_max = min(w, cx + size // 2)
        sq_y_min = max(0, cy - size // 2)
        sq_y_max = min(h, cy + size // 2)
        
        if sq_x_max - sq_x_min < 32 or sq_y_max - sq_y_min < 32:
            return None
            
        return (sq_x_min, sq_y_min, sq_x_max, sq_y_max)

    def generate_with_compositing(
        self,
        frame: np.ndarray,
        tracking_data: Optional[Any] = None,
        background_mode: str = "original"
    ) -> np.ndarray:
        """
        Generate stylized art with background handling.
        
        Args:
            frame: Input BGR image
            tracking_data: If provided with segmentation mask, only stylizes person
            background_mode: "original" to keep background, "stylize" for full frame
            
        Returns:
            Composited output image
        """
        if self.model_type == "face":
            # Face-only optimization
            bbox = self._get_face_bbox(frame, tracking_data)
            if bbox is None:
                return frame.copy()
            
            x1, y1, x2, y2 = bbox
            face_crop = frame[y1:y2, x1:x2]
            
            # Generate styled face
            stylized_face = self.generate(face_crop, None)
            
            # Composite styled face back onto the original frame with soft blending
            stylized_face_resized = cv2.resize(stylized_face, (x2 - x1, y2 - y1))
            
            result = frame.copy()
            
            # Optional: Circular soft mask for seamless edges
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
            cv2.circle(mask, ((x2 - x1)//2, (y2 - y1)//2), min((x2-x1)//2, (y2-y1)//2) - 5, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask_3ch = np.stack([mask]*3, axis=-1)
            
            result[y1:y2, x1:x2] = (stylized_face_resized * mask_3ch + face_crop * (1 - mask_3ch)).astype(np.uint8)
            return result
            
        # Full frame stylization
        stylized = self.generate(frame, tracking_data)
        
        # If we have a segmentation mask, composite the person
        if (
            tracking_data is not None 
            and hasattr(tracking_data, 'segmentation_mask')
            and tracking_data.segmentation_mask is not None
            and background_mode == "original"
        ):
            mask = tracking_data.segmentation_mask
            # Resize mask if needed
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # Normalize mask to [0, 1]
            mask_float = mask.astype(np.float32) / 255.0
            # Ensure mask is 2D before stacking
            if len(mask_float.shape) == 3:
                mask_float = mask_float.squeeze(-1)
            mask_3ch = np.stack([mask_float] * 3, axis=-1)
            
            # Composite: stylized person + original background
            composite = (stylized * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)
            return composite
        
        return stylized
    
    def close(self) -> None:
        """
        Release model resources and clear CUDA cache.
        """
        print("[ArtGenerator] Releasing resources...")
        
        # Delete model tensors
        del self.model
        del self._dummy_input
        
        # Clear cache
        self._clear_cuda_cache()
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        print("[ArtGenerator] Resources released")
    
    def __enter__(self) -> "ArtGenerator":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup."""
        self.close()


# Import cv2 for postprocess
import cv2
