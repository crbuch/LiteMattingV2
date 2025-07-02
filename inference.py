#!/usr/bin/env python3
"""
LiteMatting Inference Script

Simple script to run inference with the trained LiteMatting model.
Usage: python inference.py --input image.jpg --output alpha.png
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os

from network import MobileMatting


def generate_trimap(image, method='auto', kernel_size=10):
    """
    Generate trimap from image using different methods
    
    Args:
        image: Input image (H, W, 3)
        method: 'auto', 'center', or path to trimap file
        kernel_size: Erosion/dilation kernel size
    """
    h, w = image.shape[:2]
    
    if method == 'auto':
        # Simple automatic trimap based on image gradients
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create trimap: edges are unknown, center is foreground, borders are background
        trimap = np.ones((h, w), dtype=np.uint8) * 128  # Unknown
        
        # Set borders as background
        border_size = min(h, w) // 10
        trimap[:border_size, :] = 0
        trimap[-border_size:, :] = 0
        trimap[:, :border_size] = 0
        trimap[:, -border_size:] = 0
        
        # Set center region as foreground
        center_h, center_w = h // 2, w // 2
        fg_size = min(h, w) // 4
        trimap[center_h-fg_size:center_h+fg_size, center_w-fg_size:center_w+fg_size] = 255
        
    elif method == 'center':
        # Simple center-based trimap
        trimap = np.zeros((h, w), dtype=np.uint8)  # Background
        center_h, center_w = h // 2, w // 2
        
        # Create circular foreground region
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 <= (min(h, w) // 4)**2
        trimap[mask] = 255  # Foreground
        
        # Create unknown region around foreground
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2, kernel_size*2))
        fg_dilated = cv2.dilate((trimap == 255).astype(np.uint8), kernel, iterations=1)
        fg_eroded = cv2.erode((trimap == 255).astype(np.uint8), kernel, iterations=1)
        
        trimap[(fg_dilated == 1) & (fg_eroded == 0)] = 128  # Unknown
        
    elif os.path.exists(method):
        # Load trimap from file
        trimap = cv2.imread(method, cv2.IMREAD_GRAYSCALE)
        trimap = cv2.resize(trimap, (w, h))
        
    else:
        raise ValueError(f"Unknown trimap method: {method}")
    
    return trimap


def preprocess_inputs(image, trimap, input_size=512):
    """
    Preprocess image and trimap for model input
    
    Args:
        image: RGB image (H, W, 3)
        trimap: Grayscale trimap (H, W)
        input_size: Model input size
    
    Returns:
        Preprocessed tensors
    """
    # Resize inputs
    image_resized = cv2.resize(image, (input_size, input_size))
    trimap_resized = cv2.resize(trimap, (input_size, input_size))
    
    # Convert image to tensor
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    # Convert trimap to one-hot encoding
    trimap_onehot = np.zeros((3, input_size, input_size), dtype=np.float32)
    trimap_onehot[0] = (trimap_resized == 0).astype(np.float32)    # Background
    trimap_onehot[1] = (trimap_resized == 128).astype(np.float32)  # Unknown
    trimap_onehot[2] = (trimap_resized == 255).astype(np.float32)  # Foreground
    trimap_tensor = torch.from_numpy(trimap_onehot).unsqueeze(0)
    
    # Generate click guidance (simplified - all zeros for now)
    clicks = np.zeros((2, input_size, input_size), dtype=np.float32)
    clicks_tensor = torch.from_numpy(clicks).unsqueeze(0)
    
    return image_tensor, trimap_tensor, clicks_tensor


def postprocess_alpha(alpha_pred, original_shape):
    """
    Postprocess predicted alpha matte
    
    Args:
        alpha_pred: Predicted alpha (1, 1, H, W)
        original_shape: (H, W) of original image
    
    Returns:
        Alpha matte resized to original shape
    """
    alpha = alpha_pred.squeeze().cpu().numpy()
    alpha = cv2.resize(alpha, (original_shape[1], original_shape[0]))
    alpha = np.clip(alpha, 0, 1)
    return alpha


def run_inference(model, image_path, trimap_method='auto', output_path=None, device='cpu'):
    """
    Run inference on a single image
    
    Args:
        model: Trained MobileMatting model
        image_path: Path to input image
        trimap_method: Method to generate trimap
        output_path: Path to save alpha matte
        device: Device to run inference on
    
    Returns:
        Alpha matte as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Generate trimap
    trimap = generate_trimap(image, trimap_method)
    
    # Preprocess inputs
    image_tensor, trimap_tensor, clicks_tensor = preprocess_inputs(image, trimap)
    image_tensor = image_tensor.to(device)
    trimap_tensor = trimap_tensor.to(device)
    clicks_tensor = clicks_tensor.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        alpha_pred = model(image_tensor, trimap_tensor, clicks_tensor)
    
    # Postprocess alpha
    alpha = postprocess_alpha(alpha_pred, original_shape)
    
    # Save alpha matte
    if output_path:
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        cv2.imwrite(output_path, alpha_uint8)
        print(f"Alpha matte saved to: {output_path}")
    
    return alpha, trimap


def create_composite(image, alpha, background_color=(0, 255, 0)):
    """
    Create composite image with new background
    
    Args:
        image: Original image (H, W, 3)
        alpha: Alpha matte (H, W)
        background_color: RGB background color
    
    Returns:
        Composite image
    """
    alpha_3ch = alpha[..., np.newaxis]
    bg_color = np.array(background_color, dtype=np.float32) / 255.0
    
    composite = alpha_3ch * (image / 255.0) + (1 - alpha_3ch) * bg_color
    composite = (composite * 255).astype(np.uint8)
    
    return composite


def main():
    parser = argparse.ArgumentParser(description='LiteMatting Inference')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', help='Output alpha matte path')
    parser.add_argument('--model', '-m', default='checkpoints/best_checkpoint.pth', 
                       help='Path to trained model checkpoint')
    parser.add_argument('--trimap', '-t', default='auto', 
                       help='Trimap method: auto, center, or path to trimap file')
    parser.add_argument('--composite', '-c', help='Save composite image with new background')
    parser.add_argument('--background', '-b', nargs=3, type=int, default=[0, 255, 0],
                       help='Background color (R G B) for composite')
    parser.add_argument('--device', '-d', default='auto', 
                       help='Device to use: auto, cpu, cuda')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model = MobileMatting().to(device)
    
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from: {args.model}")
    else:
        print(f"Warning: Model file not found: {args.model}")
        print("Using randomly initialized model (will produce poor results)")
    
    # Set output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_alpha.png"
    
    # Run inference
    print(f"Processing: {args.input}")
    alpha, trimap = run_inference(model, args.input, args.trimap, args.output, device)
    
    # Create composite if requested
    if args.composite:
        image = cv2.imread(args.input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        composite = create_composite(image, alpha, args.background)
        composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.composite, composite_bgr)
        print(f"Composite saved to: {args.composite}")
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
