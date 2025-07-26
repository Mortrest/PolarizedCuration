#!/usr/bin/env python3
"""
Example usage of the CIFAR-10 flow matching model.

This script demonstrates how to:
1. Load different pre-trained models
2. Generate samples with different parameters
3. Save individual samples
4. Compare different integration methods
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the conditional-flow-matching directory to the path
sys.path.append('./conditional-flow-matching')

from use_pretrained_model import load_model, generate_samples, save_samples


def compare_integration_methods():
    """Compare different ODE integration methods."""
    print("Comparing integration methods...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("otcfm", step=400000, device=device)
    
    methods = ["dopri5", "euler", "rk4"]
    samples_per_method = 4
    
    fig, axes = plt.subplots(len(methods), samples_per_method, figsize=(12, 9))
    
    for i, method in enumerate(methods):
        print(f"Generating samples with {method}...")
        samples = generate_samples(
            model, 
            num_samples=samples_per_method, 
            integration_method=method,
            device=device
        )
        
        for j in range(samples_per_method):
            img = samples[j].transpose(1, 2, 0)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(method, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("integration_methods_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison to integration_methods_comparison.png")


def generate_high_quality_samples():
    """Generate high-quality samples using the best settings."""
    print("Generating high-quality samples...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("otcfm", step=400000, device=device)
    
    # Generate more samples with high-quality settings
    samples = generate_samples(
        model,
        num_samples=64,
        batch_size=8,
        integration_method="dopri5",
        integration_steps=200,  # More steps for better quality
        device=device
    )
    
    # Save as a larger grid
    save_samples(samples, filename="high_quality_samples.png")
    
    # Also save individual samples
    output_dir = Path("./individual_samples")
    output_dir.mkdir(exist_ok=True)
    
    for i in range(min(10, len(samples))):  # Save first 10 samples
        plt.figure(figsize=(4, 4))
        img = samples[i].transpose(1, 2, 0)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(output_dir / f"sample_{i:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved individual samples to {output_dir}")


def compare_model_types():
    """Compare different model types (OT-CFM vs I-CFM)."""
    print("Comparing model types...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models = {
        "OT-CFM": "otcfm",
        "I-CFM": "icfm"
    }
    
    samples_per_model = 8
    fig, axes = plt.subplots(len(models), samples_per_model, figsize=(16, 4))
    
    for i, (name, model_type) in enumerate(models.items()):
        print(f"Loading {name} model...")
        try:
            model = load_model(model_type, step=400000, device=device)
            
            samples = generate_samples(
                model,
                num_samples=samples_per_model,
                integration_method="dopri5",
                device=device
            )
            
            for j in range(samples_per_model):
                img = samples[j].transpose(1, 2, 0)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(name, fontsize=12)
                    
        except Exception as e:
            print(f"Failed to load {name} model: {e}")
            for j in range(samples_per_model):
                axes[i, j].text(0.5, 0.5, f"{name}\nNot Available", 
                              ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved model comparison to model_comparison.png")


def demonstrate_sampling_process():
    """Demonstrate the sampling process step by step."""
    print("Demonstrating sampling process...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("otcfm", step=400000, device=device)
    
    # Generate trajectory
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32, device=device)
        
        # Sample at different time steps
        time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
        trajectory = []
        
        for t in time_steps:
            t_tensor = torch.ones(1, device=device) * t
            velocity = model(t_tensor, x)
            trajectory.append(x.clone().cpu().numpy())
            
            if t < 1.0:
                dt = 0.25
                x = x + velocity * dt
    
    # Visualize trajectory
    fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 3))
    
    for i, (t, sample) in enumerate(zip(time_steps, trajectory)):
        img = sample[0].transpose(1, 2, 0)
        img = (img * 127.5 + 128).clip(0, 255).astype(np.uint8)
        
        axes[i].imshow(img)
        axes[i].set_title(f"t = {t:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("sampling_trajectory.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved sampling trajectory to sampling_trajectory.png")


def main():
    """Run all demonstrations."""
    print("CIFAR-10 Flow Matching Model Demonstrations")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run demonstrations
    try:
        compare_integration_methods()
        generate_high_quality_samples()
        compare_model_types()
        demonstrate_sampling_process()
        
        print("\nAll demonstrations completed successfully!")
        print("Check the generated images in the current directory.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 