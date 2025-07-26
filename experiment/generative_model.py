"""
Generative Model using Optimal Transport Conditional Flow Matching (OT-CFM).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Add the conditional-flow-matching directory to the path
sys.path.append('../conditional-flow-matching')

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper


class OTFlowModel(nn.Module):
    """
    Optimal Transport Conditional Flow Matching model for CIFAR-10.
    """
    
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        hidden_size: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.device = device
        
        # Initialize the UNet backbone for the flow model
        self.unet = UNetModelWrapper(
            dim=(in_channels, image_size, image_size),
            num_res_blocks=2,
            num_channels=hidden_size,
            channel_mult=[1, 2, 2, 2],
            num_heads=num_heads,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=dropout,
        )
        
        # Initialize the Conditional Flow Matcher
        self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(
            sigma=0.0  # No noise for optimal transport
        )
        
        self.to(device)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, args=None) -> torch.Tensor:
        """
        Forward pass of the flow model.
        
        Args:
            t: Time tensor of shape (batch_size,)
            x: Input tensor of shape (batch_size, channels, height, width)
            args: Additional arguments (ignored for compatibility)
            
        Returns:
            Predicted velocity field
        """
        return self.unet(t, x)
    
    def sample(
        self,
        batch_size: int = 1,
        num_steps: int = 100,
        integration_method: str = "dopri5",
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using the trained flow model.
        
        Args:
            batch_size: Number of samples to generate
            num_steps: Number of integration steps for sampling
            integration_method: ODE solver method
            return_trajectory: Whether to return the full sampling trajectory
            
        Returns:
            Generated samples of shape (batch_size, channels, height, width)
        """
        self.eval()
        
        # Start from noise
        x = torch.randn(
            batch_size, self.in_channels, self.image_size, self.image_size,
            device=self.device
        )
        
        if return_trajectory:
            trajectory = [x.clone()]
        
        # Integration for sampling
        if integration_method == "euler":
            node = NeuralODE(self, solver=integration_method)
            t_span = torch.linspace(0, 1, num_steps + 1, device=self.device)
            traj = node.trajectory(x, t_span=t_span)
            sample = traj[-1]
        else:
            t_span = torch.linspace(0, 1, 2, device=self.device)
            with torch.no_grad():
                traj = odeint(
                    self, x, t_span, rtol=1e-5, atol=1e-5, method=integration_method
                )
            sample = traj[-1]
            
            if return_trajectory:
                trajectory = traj
        
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        else:
            return sample
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional flow matching loss.
        
        Args:
            x0: Initial samples
            x1: Target samples  
            t: Time points
            
        Returns:
            Loss value
        """
        # Get the conditional trajectory
        xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1, t)
        
        # Predict the velocity field
        predicted_velocity = self.forward(t, xt)
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_velocity, ut, reduction='mean')
        
        return loss
    
    def save_model(self, path: str):
        """Save the model to disk."""
        # Handle different UNet types
        if hasattr(self.unet, 'num_channels'):
            hidden_size = self.unet.num_channels
        elif hasattr(self.unet, 'model_channels'):
            hidden_size = self.unet.model_channels
        else:
            hidden_size = 128  # Default fallback
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'image_size': self.image_size,
                'in_channels': self.in_channels,
                'hidden_size': hidden_size,
                'num_blocks': 2,  # Fixed in UNetModelWrapper
                'num_heads': getattr(self.unet, 'num_heads', 8),
                'dropout': getattr(self.unet, 'dropout', 0.1)
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = None) -> 'OTFlowModel':
        """Load a saved model from disk."""
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if 'config' in checkpoint:
            # Our custom format
            config = checkpoint['config']
            model = cls(
                image_size=config['image_size'],
                in_channels=config['in_channels'],
                hidden_size=config['hidden_size'],
                num_blocks=config['num_blocks'],
                num_heads=config['num_heads'],
                dropout=config['dropout'],
                device=device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'net_model' in checkpoint:
            # Pre-trained format from conditional-flow-matching
            # The pre-trained model has a different UNet architecture
            # We need to use the exact same architecture as the pre-trained model
            import sys
            sys.path.append('../conditional-flow-matching/torchcfm')
            from models.unet.unet import UNetModelWrapper
            
            # Create the model using the exact same architecture as pre-trained
            model = UNetModelWrapper(
                dim=(3, 32, 32),
                num_res_blocks=2,
                num_channels=128,  # FLAGS.num_channel
                channel_mult=[1, 2, 2, 2],
                num_heads=4,
                num_head_channels=64,
                attention_resolutions="16",
                dropout=0.1,
            )
            
            # Load the pre-trained weights
            model.load_state_dict(checkpoint['net_model'])
            
            # Wrap it in our OTFlowModel interface
            ot_model = cls(
                image_size=32,
                in_channels=3,
                hidden_size=128,
                num_blocks=2,
                num_heads=8,
                dropout=0.1,
                device=device
            )
            
            # Replace the UNet with the pre-trained one
            ot_model.unet = model.to(device)
            return ot_model
        else:
            raise ValueError(f"Unknown checkpoint format: {list(checkpoint.keys())}")
        
        return model


class FlowTrainer:
    """
    Trainer for the OT-CFM model.
    """
    
    def __init__(
        self,
        model: OTFlowModel,
        learning_rate: float = 2e-4,
        warmup_steps: int = 5000,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        def warmup_lr(step):
            return min(step, warmup_steps) / warmup_steps
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_lr
        )
    
    def train_step(
        self,
        x1: torch.Tensor,
        x0: Optional[torch.Tensor] = None
    ) -> float:
        """
        Single training step.
        
        Args:
            x1: Target samples
            x0: Initial samples (if None, will be sampled from noise)
            
        Returns:
            Loss value
        """
        if x0 is None:
            x0 = torch.randn_like(x1)
        
        # Sample from flow matching
        t, xt, ut = self.model.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        
        # Forward pass
        vt = self.model(t, xt)
        
        # Compute loss
        loss = torch.mean((vt - ut) ** 2)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train_on_dataset(
        self,
        dataloader,
        num_steps: int,
        log_interval: int = 1000
    ) -> list:
        """
        Train the model on a dataset.
        
        Args:
            dataloader: DataLoader for training data
            num_steps: Number of training steps
            log_interval: Interval for logging
            
        Returns:
            List of loss values
        """
        self.model.train()
        losses = []
        
        # Create infinite loop from dataloader
        data_iter = iter(dataloader)
        
        # Add progress bar
        from tqdm import tqdm
        pbar = tqdm(range(num_steps), desc="Training", unit="step")
        
        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            x1 = batch[0].to(self.device)
            loss = self.train_step(x1)
            losses.append(loss)
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss:.4f}'})
            
            if step % log_interval == 0:
                print(f"Step {step}/{num_steps}, Loss: {loss:.4f}")
        
        return losses 