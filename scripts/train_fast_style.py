import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_engine import LightweightFaceStyleNet, TransformerNet

class VGGFeatureExtractor(nn.Module):
    """Extracts features from specific layers of VGG16 for perceptual loss."""
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3

def gram_matrix(y):
    """Computes Gram matrix for style loss."""
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def load_image(filename, size=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return transform(img).unsqueeze(0)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    print(f"Loading dataset from {args.dataset}")
    # Note: dataset path should point to a directory containing subdirectories of images
    try:
        train_dataset = datasets.ImageFolder(args.dataset, transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your dataset path has subdirectories containing images (e.g. data/faces/img1.jpg).")
        return
        
    # Init model
    if args.model_type == 'face':
        print("Initializing LightweightFaceStyleNet (Optimized for CPU/Face crops)")
        style_model = LightweightFaceStyleNet().to(device)
    else:
        print("Initializing Full TransformerNet")
        style_model = TransformerNet().to(device)
        
    optimizer = optim.Adam(style_model.parameters(), args.lr)
    
    # Init VGG for perceptual loss
    vgg = VGGFeatureExtractor().to(device).eval()
    
    # Load and process style image
    style_img = load_image(args.style_image, size=args.style_size).to(device)
    # VGG expects normalization roughly in range [0, 1] shifted by ImageNet mean, but here our tensors are [0, 255]
    # We apply same scale to VGG features
    style_features = vgg(style_img)
    style_gram = [gram_matrix(x) for x in style_features]
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Starting training for {args.epochs} epochs over {len(train_loader)} batches...")
    
    for epoch in range(args.epochs):
        style_model.train()
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            x = x.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Output is [0, 1], so we multiply by 255 to match input expectation for perceptual loss here
            y = style_model(x) * 255.0
            
            # VGG Features
            x_features = vgg(x)
            y_features = vgg(y)
            
            # Content Loss (relu2_2)
            content_loss = args.content_weight * F.mse_loss(y_features[1], x_features[1])
            
            # Style Loss
            style_loss = 0.
            for ft_y, gm_s in zip(y_features, style_gram):
                gm_y = gram_matrix(ft_y)
                style_loss += F.mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight
            
            # Total Variation (Smoothness) Loss
            tv_loss = args.tv_weight * (
                torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
            ) / (n_batch * 3 * args.image_size * args.image_size)
            
            total_loss = content_loss + style_loss + tv_loss
            
            total_loss.backward()
            optimizer.step()
            
            if (batch_id + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_id+1}/{len(train_loader)}] "
                      f"Content: {content_loss.item():.2f} Style: {style_loss.item():.2f} "
                      f"TV: {tv_loss.item():.6f} Total: {total_loss.item():.2f}")
                
        # Save checkpoint end of epoch
        ckpt_path = os.path.join(args.save_dir, f"style_model_{args.model_type}_epoch_{epoch+1}.pth")
        torch.save(style_model.state_dict(), ckpt_path)
        print(f"Saved {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fast Neural Style Transfer Model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset folder")
    parser.add_argument("--style-image", type=str, required=True, help="Path to style reference image")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save weights")
    parser.add_argument("--model-type", choices=['full', 'face'], default='face', help="Model architecture")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128, help="Training crop size (128 for face, 256 for full)")
    parser.add_argument("--style-size", type=int, default=None, help="Size to resize style image")
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # Loss weights
    parser.add_argument("--content-weight", type=float, default=1e5)
    parser.add_argument("--style-weight", type=float, default=1e10)
    parser.add_argument("--tv-weight", type=float, default=1e-5)
    
    parser.add_argument("--log-interval", type=int, default=50)
    
    args = parser.parse_args()
    train(args)
