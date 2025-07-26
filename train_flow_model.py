#!/usr/bin/env python3
"""
Simplified training script for CIFAR-10 flow matching models.

This script is based on the original training code from:
https://github.com/atong01/conditional-flow-matching/tree/main/examples/images/cifar10

Usage:
    python train_flow_model.py --model otcfm --total_steps 100000 --batch_size 64
"""

import os
import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

# Add the conditional-flow-matching directory to the path
sys.path.append('./conditional-flow-matching')

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper


def ema(model, ema_model, decay):
    """Exponential moving average update."""
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def infiniteloop(dataloader):
    """Create an infinite loop from a dataloader."""
    while True:
        for batch in dataloader:
            yield batch


def train_flow_model(args):
    """Train a flow matching model on CIFAR-10."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data loading
    print("Loading CIFAR-10 dataset...")
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    
    datalooper = infiniteloop(dataloader)
    
    # Model initialization
    print("Initializing model...")
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=args.num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    
    ema_model = copy.deepcopy(net_model)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    
    def warmup_lr(step):
        return min(step, args.warmup) / args.warmup
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    
    # Print model size
    model_size = sum(param.data.nelement() for param in net_model.parameters())
    print(f"Model parameters: {model_size / 1024 / 1024:.2f} M")
    
    # Flow matching setup
    sigma = 0.0
    if args.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif args.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif args.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif args.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError(f"Unknown model {args.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']")
    
    print(f"Using {args.model} flow matching")
    
    # Training loop
    print(f"Starting training for {args.total_steps} steps...")
    progress_bar = tqdm(range(args.total_steps), desc="Training")
    
    for step in progress_bar:
        optimizer.zero_grad()
        
        # Get batch
        x1 = next(datalooper).to(device)
        x0 = torch.randn_like(x1)
        
        # Sample from flow matching
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        
        # Forward pass
        vt = net_model(t, xt)
        
        # Compute loss
        loss = torch.mean((vt - ut) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Update EMA
        ema(net_model, ema_model, args.ema_decay)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # Save checkpoint
        if args.save_step > 0 and step % args.save_step == 0:
            checkpoint_path = os.path.join(
                args.output_dir, 
                f"{args.model}_cifar10_weights_step_{step}.pt"
            )
            
            torch.save({
                "net_model": net_model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "args": vars(args)
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 flow matching model")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="otcfm", 
                       choices=["otcfm", "icfm", "fm", "si"],
                       help="Flow matching model type")
    parser.add_argument("--num_channels", type=int, default=128,
                       help="Base number of channels in UNet")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--total_steps", type=int, default=400001, help="Total training steps")
    parser.add_argument("--warmup", type=int, default=5000, help="Learning rate warmup steps")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for checkpoints")
    parser.add_argument("--save_step", type=int, default=20000,
                       help="Frequency of saving checkpoints")
    
    args = parser.parse_args()
    
    train_flow_model(args)


if __name__ == "__main__":
    main() 