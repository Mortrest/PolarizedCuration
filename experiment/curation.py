"""
K-Choice Curation Mechanism for filtering synthetic samples.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import random


class KChoiceCurator:
    """
    K-Choice curation mechanism based on Bradley-Terry model.
    """
    
    def __init__(self, k: int = 4, device: str = "cuda"):
        """
        Initialize the K-Choice curator.
        
        Args:
            k: Number of samples to group together for selection
            device: Device to use for computations
        """
        self.k = k
        self.device = device
    
    def curate_samples(
        self,
        samples: torch.Tensor,
        rewards: torch.Tensor,
        num_curated: int,
        batch_size: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Curate samples using K-choice filtering.
        
        Args:
            samples: Synthetic samples of shape (num_samples, 3, 32, 32)
            rewards: Rewards for each sample of shape (num_samples,)
            num_curated: Number of samples to curate
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (curated_samples, curated_rewards)
        """
        print(f"Curating {num_curated} samples from {len(samples)} using K={self.k} choice...")
        
        curated_samples = []
        curated_rewards = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            batch_rewards = rewards[i:i + batch_size]
            
            batch_curated_samples, batch_curated_rewards = self._curate_batch(
                batch_samples, batch_rewards, num_curated // (len(samples) // batch_size + 1)
            )
            
            curated_samples.append(batch_curated_samples)
            curated_rewards.append(batch_curated_rewards)
        
        # Concatenate all curated samples
        if curated_samples:
            curated_samples = torch.cat(curated_samples, dim=0)
            curated_rewards = torch.cat(curated_rewards, dim=0)
            
            # If we have more than needed, randomly sample
            if len(curated_samples) > num_curated:
                indices = torch.randperm(len(curated_samples))[:num_curated]
                curated_samples = curated_samples[indices]
                curated_rewards = curated_rewards[indices]
        
        print(f"Curated {len(curated_samples)} samples")
        return curated_samples, curated_rewards
    
    def _curate_batch(
        self,
        samples: torch.Tensor,
        rewards: torch.Tensor,
        num_curated: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Curate samples from a single batch.
        
        Args:
            samples: Batch of samples
            rewards: Batch of rewards
            num_curated: Number of samples to curate
            
        Returns:
            Tuple of (curated_samples, curated_rewards)
        """
        if len(samples) < self.k:
            # If not enough samples, return all
            return samples, rewards
        
        curated_samples = []
        curated_rewards = []
        
        # Randomly shuffle indices
        indices = torch.randperm(len(samples))
        
        # Group samples into batches of K
        num_groups = len(samples) // self.k
        
        for i in range(min(num_groups, num_curated)):
            # Get K samples for this group
            group_indices = indices[i * self.k:(i + 1) * self.k]
            group_samples = samples[group_indices]
            group_rewards = rewards[group_indices]
            
            # Select one sample using softmax over rewards
            selected_idx = self._select_sample(group_rewards)
            selected_sample = group_samples[selected_idx]
            selected_reward = group_rewards[selected_idx]
            
            curated_samples.append(selected_sample)
            curated_rewards.append(selected_reward)
        
        if curated_samples:
            return torch.stack(curated_samples), torch.stack(curated_rewards)
        else:
            return torch.empty(0, *samples.shape[1:], device=self.device), torch.empty(0, device=self.device)
    
    def _select_sample(self, rewards: torch.Tensor) -> int:
        """
        Select one sample from a group using softmax over rewards.
        
        Args:
            rewards: Rewards for K samples
            
        Returns:
            Index of selected sample
        """
        # Compute softmax probabilities
        logits = rewards / 0.1  # Temperature parameter
        probabilities = torch.softmax(logits, dim=0)
        
        # Sample from the distribution
        selected_idx = torch.multinomial(probabilities, 1).item()
        
        return selected_idx
    
    def get_curation_stats(
        self,
        samples: torch.Tensor,
        rewards: torch.Tensor,
        num_curated: int
    ) -> dict:
        """
        Get statistics about the curation process.
        
        Args:
            samples: Original samples
            rewards: Original rewards
            num_curated: Number of curated samples
            
        Returns:
            Dictionary with curation statistics
        """
        original_mean_reward = rewards.mean().item()
        original_std_reward = rewards.std().item()
        original_min_reward = rewards.min().item()
        original_max_reward = rewards.max().item()
        
        curated_samples, curated_rewards = self.curate_samples(
            samples, rewards, num_curated
        )
        
        if len(curated_rewards) > 0:
            curated_mean_reward = curated_rewards.mean().item()
            curated_std_reward = curated_rewards.std().item()
            curated_min_reward = curated_rewards.min().item()
            curated_max_reward = curated_rewards.max().item()
        else:
            curated_mean_reward = curated_std_reward = curated_min_reward = curated_max_reward = 0.0
        
        stats = {
            "original_mean_reward": original_mean_reward,
            "original_std_reward": original_std_reward,
            "original_min_reward": original_min_reward,
            "original_max_reward": original_max_reward,
            "curated_mean_reward": curated_mean_reward,
            "curated_std_reward": curated_std_reward,
            "curated_min_reward": curated_min_reward,
            "curated_max_reward": curated_max_reward,
            "reward_improvement": curated_mean_reward - original_mean_reward,
            "num_original": len(samples),
            "num_curated": len(curated_samples),
            "curation_ratio": len(curated_samples) / len(samples) if len(samples) > 0 else 0.0
        }
        
        return stats


class RewardComputer:
    """
    Helper class for computing rewards using the classifier.
    """
    
    def __init__(self, classifier, device: str = "cuda"):
        self.classifier = classifier
        self.device = device
    
    def compute_rewards(
        self,
        samples: torch.Tensor,
        reward_type: str = "confidence",
        target_class: int = 0,
        gamma: float = 5.0
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of samples.
        
        Args:
            samples: Batch of samples
            reward_type: Type of reward function
            target_class: Target class for single_class reward
            gamma: Scaling factor
            
        Returns:
            Rewards tensor
        """
        return self.classifier.get_rewards(
            samples, reward_type, target_class, gamma
        )
    
    def get_class_distribution(self, samples: torch.Tensor) -> torch.Tensor:
        """Get class distribution for samples."""
        return self.classifier.get_class_distribution(samples) 