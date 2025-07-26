"""
VGG11 Classifier for reward computation in the retraining experiment.
"""

import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional, Tuple, List

# Add the PyTorch_CIFAR10 directory to the path
sys.path.append('../PyTorch_CIFAR10')

from cifar10_models.vgg import vgg11_bn


class VGG11Classifier:
    """
    VGG11 classifier wrapper for computing rewards in the retraining experiment.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.transform = None
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained VGG11 model."""
        print("Loading pre-trained VGG11 classifier...")
        
        # Load the pre-trained model
        self.model = vgg11_bn(pretrained=True, device="cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define the normalization transform
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        print(f"VGG11 classifier loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_class_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities for a batch of images.
        
        Args:
            images: Batch of images of shape (batch_size, 3, 32, 32)
                   Expected to be in range [0, 1]
        
        Returns:
            Class probabilities of shape (batch_size, 10)
        """
        # Apply normalization if images are in [0, 1] range
        if images.max() <= 1.0:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1).to(self.device)
            images = (images - mean) / std
        
        with torch.no_grad():
            logits = self.model(images)
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities
    
    def get_rewards(
        self,
        images: torch.Tensor,
        reward_type: str = "confidence",
        target_class: int = 0,
        gamma: float = 5.0
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of images.
        
        Args:
            images: Batch of images of shape (batch_size, 3, 32, 32)
            reward_type: Type of reward function ("confidence" or "single_class")
            target_class: Target class for single_class reward (default: 0 for airplane)
            gamma: Scaling factor for rewards
            
        Returns:
            Rewards of shape (batch_size,)
        """
        probabilities = self.get_class_probabilities(images)
        
        if reward_type == "confidence":
            # Reward based on maximum class confidence
            max_probs = torch.max(probabilities, dim=1)[0]
            rewards = gamma * max_probs
        
        elif reward_type == "single_class":
            # Reward based on target class probability
            target_probs = probabilities[:, target_class]
            rewards = gamma * target_probs
        
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        return rewards
    
    def get_class_distribution(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get the class distribution for a batch of images.
        
        Args:
            images: Batch of images of shape (batch_size, 3, 32, 32)
        
        Returns:
            Class distribution of shape (10,) - counts for each class
        """
        probabilities = self.get_class_probabilities(images)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        # Count occurrences of each class
        class_counts = torch.zeros(10, device=self.device)
        for i in range(10):
            class_counts[i] = (predicted_classes == i).sum()
        
        return class_counts
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return self.class_names.copy()
    
    def get_accuracy(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute accuracy on a batch of images.
        
        Args:
            images: Batch of images of shape (batch_size, 3, 32, 32)
            labels: Ground truth labels of shape (batch_size,)
        
        Returns:
            Accuracy as a float
        """
        probabilities = self.get_class_probabilities(images)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        correct = (predicted_classes == labels).sum().item()
        total = labels.size(0)
        
        return correct / total if total > 0 else 0.0 