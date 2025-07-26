#!/usr/bin/env python3
"""
Script to use the pre-trained CIFAR-10 flow matching model.

This script downloads the pre-trained weights and generates samples using the
Optimal Transport Conditional Flow Matching (OT-CFM) model trained on CIFAR-10.

Based on the repository: https://github.com/atong01/conditional-flow-matching
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
import urllib.request
from pathlib import Path

# Add the conditional-flow-matching directory to the path
sys.path.append('./conditional-flow-matching')

from torchcfm.models.unet.unet import UNetModelWrapper


def download_pretrained_weights(model_type="otcfm", step=400000):
    """
    Download pre-trained weights from the official repository.
    
    Args:
        model_type: Type of model ('otcfm', 'icfm', 'fm')
        step: Training step number
    """
    weights_dir = Path("./pretrained_weights")
    weights_dir.mkdir(exist_ok=True)
    
    weights_path = weights_dir / f"{model_type}_cifar10_weights_step_{step}.pt"
    
    if not weights_path.exists():
        print(f"Downloading {model_type} weights...")
        url = f"https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/{model_type}_cifar10_weights_step_{step}.pt"
        
        try:
            urllib.request.urlretrieve(url, weights_path)
            print(f"Downloaded weights to {weights_path}")
        except Exception as e:
            print(f"Failed to download weights: {e}")
            print("Please download manually from:")
            print(f"https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/{model_type}_cifar10_weights_step_{step}.pt")
            return None
    
    return weights_path


def load_model(model_type="otcfm", step=400000, device="cuda"):
    """
    Load the pre-trained model.
    
    Args:
        model_type: Type of model ('otcfm', 'icfm', 'fm')
        step: Training step number
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    # Download weights if not present
    weights_path = download_pretrained_weights(model_type, step)
    if weights_path is None:
        return None
    
    # Define the model architecture
    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    
    # Load the weights
    print(f"Loading weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Load EMA model weights (better quality)
    state_dict = checkpoint["ema_model"]
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Handle DataParallel wrapper
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        model.load_state_dict(new_state_dict)
    
    model.eval()
    print(f"Successfully loaded {model_type} model")
    
    return model


def generate_samples(model, num_samples=16, batch_size=4, integration_method="dopri5", 
                    integration_steps=100, device="cuda"):
    """
    Generate samples using the trained model.
    
    Args:
        model: The trained model
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        integration_method: ODE solver method ('dopri5', 'euler', etc.)
        integration_steps: Number of integration steps for Euler method
        device: Device to use
    
    Returns:
        Generated samples as numpy array
    """
    samples = []
    
    # Define the integration method
    if integration_method == "euler":
        node = NeuralODE(model, solver=integration_method)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Start from noise
            x = torch.randn(current_batch_size, 3, 32, 32, device=device)
            
            if integration_method == "euler":
                t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
                traj = node.trajectory(x, t_span=t_span)
                sample = traj[-1]
            else:
                t_span = torch.linspace(0, 1, 2, device=device)
                traj = odeint(model, x, t_span, rtol=1e-5, atol=1e-5, method=integration_method)
                sample = traj[-1]
            
            # Convert to image format
            sample = (sample * 127.5 + 128).clip(0, 255).to(torch.uint8)
            samples.append(sample.cpu().numpy())
    
    return np.concatenate(samples, axis=0)


def save_samples(samples, output_dir="./generated_samples", filename="cifar10_samples.png"):
    """
    Save generated samples as images.
    
    Args:
        samples: Generated samples as numpy array
        output_dir: Directory to save samples
        filename: Filename for the grid image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a grid of images
    n_samples = len(samples)
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        if i < n_samples:
            # Convert from (C, H, W) to (H, W, C) for matplotlib
            img = samples[i].transpose(1, 2, 0)
            axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {n_samples} samples to {os.path.join(output_dir, filename)}")


def main():
    """Main function to demonstrate the pre-trained model usage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the pre-trained OT-CFM model
    model = load_model(model_type="otcfm", step=400000, device=device)
    
    if model is None:
        print("Failed to load model. Please check the weights file.")
        return
    
    # Generate samples
    print("Generating samples...")
    samples = generate_samples(
        model, 
        num_samples=16, 
        batch_size=4, 
        integration_method="dopri5",
        device=device
    )
    
    # Save samples
    save_samples(samples, filename="otcfm_cifar10_samples.png")
    
    print("Done! Check the generated_samples directory for the output images.")


if __name__ == "__main__":
    main() 