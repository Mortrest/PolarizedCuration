"""
Data loader for CIFAR-10 dataset.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Tuple, Optional


class CIFAR10DataLoader:
    """
    Data loader for CIFAR-10 dataset with proper preprocessing.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        device: str = "cuda"
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        # CIFAR-10 normalization values
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2471, 0.2435, 0.2616]
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load CIFAR-10 datasets."""
        print("Loading CIFAR-10 datasets...")
        
        # Training dataset
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Test dataset
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
    
    def get_train_subset(self, num_samples: int) -> DataLoader:
        """Get a subset of training data."""
        indices = torch.randperm(len(self.train_dataset))[:num_samples]
        subset = Subset(self.train_dataset, indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
    
    def get_class_balanced_subset(self, samples_per_class: int) -> DataLoader:
        """Get a class-balanced subset of training data."""
        class_indices = [[] for _ in range(10)]
        
        # Collect indices for each class
        for idx, (_, label) in enumerate(self.train_dataset):
            class_indices[label].append(idx)
        
        # Sample from each class
        selected_indices = []
        for class_idx in range(10):
            class_samples = class_indices[class_idx]
            if len(class_samples) >= samples_per_class:
                selected = np.random.choice(class_samples, samples_per_class, replace=False)
            else:
                selected = np.random.choice(class_samples, samples_per_class, replace=True)
            selected_indices.extend(selected)
        
        subset = Subset(self.train_dataset, selected_indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
    
    def get_class_samples(self, class_idx: int, num_samples: int) -> torch.Tensor:
        """Get samples from a specific class."""
        class_indices = []
        
        for idx, (_, label) in enumerate(self.train_dataset):
            if label == class_idx:
                class_indices.append(idx)
                if len(class_indices) >= num_samples:
                    break
        
        if len(class_indices) < num_samples:
            # If not enough samples, sample with replacement
            class_indices = np.random.choice(class_indices, num_samples, replace=True)
        
        samples = []
        for idx in class_indices:
            sample, _ = self.train_dataset[idx]
            samples.append(sample)
        
        return torch.stack(samples)
    
    def get_class_distribution(self) -> np.ndarray:
        """Get the class distribution of the training dataset."""
        class_counts = np.zeros(10)
        
        for _, label in self.train_dataset:
            class_counts[label] += 1
        
        return class_counts
    
    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images back to [0, 1] range."""
        mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(self.std).view(1, 3, 1, 1).to(images.device)
        
        return images * std + mean


class SyntheticDataset(Dataset):
    """
    Dataset wrapper for synthetic samples.
    """
    
    def __init__(self, samples: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # Move to CPU for DataLoader compatibility
        self.samples = samples.cpu()
        self.labels = labels.cpu() if labels is not None else torch.zeros(len(samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.samples[idx], self.labels[idx]
        else:
            return self.samples[idx], 0  # Dummy label


def create_mixed_dataloader(
    real_samples: torch.Tensor,
    synthetic_samples: torch.Tensor,
    batch_size: int = 128,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a data loader that mixes real and synthetic samples.
    
    Args:
        real_samples: Real samples from CIFAR-10
        synthetic_samples: Synthetic samples from the generative model
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        DataLoader with mixed samples
    """
    # Move both tensors to CPU for DataLoader compatibility
    real_samples = real_samples.cpu()
    synthetic_samples = synthetic_samples.cpu()
    
    # Combine samples
    all_samples = torch.cat([real_samples, synthetic_samples], dim=0)
    
    # Create dummy labels (not used for training the generative model)
    dummy_labels = torch.zeros(len(all_samples))
    
    # Create dataset
    dataset = SyntheticDataset(all_samples, dummy_labels)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    return dataloader 