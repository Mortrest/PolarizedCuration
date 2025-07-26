#!/usr/bin/env python3
"""
Script to use the pre-trained VGG11 model on CIFAR-10.

This script loads the pre-trained VGG11 model from the PyTorch_CIFAR10 repository
and demonstrates how to use it for inference.

Based on: https://github.com/huyvnphan/PyTorch_CIFAR10
"""

import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add the PyTorch_CIFAR10 directory to the path
sys.path.append('./PyTorch_CIFAR10')

from cifar10_models.vgg import vgg11_bn


def load_pretrained_vgg11(device="cuda"):
    """
    Load the pre-trained VGG11 model.
    
    Args:
        device: Device to load the model on
        
    Returns:
        Loaded VGG11 model
    """
    print("Loading pre-trained VGG11 model...")
    
    # Load the pre-trained model on CPU first, then move to target device
    model = vgg11_bn(pretrained=True, device="cpu")
    model = model.to(device)
    model.eval()
    
    print(f"VGG11 model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def get_cifar10_transforms():
    """
    Get the standard CIFAR-10 transforms used during training.
    
    Returns:
        transform: Transform for test images
    """
    # CIFAR-10 normalization values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transform


def predict_single_image(model, image_path, device="cuda"):
    """
    Predict the class of a single image.
    
    Args:
        model: The trained model
        image_path: Path to the image file
        device: Device to use
        
    Returns:
        predicted_class: Predicted class index
        confidence: Confidence score
        class_name: Class name
    """
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Load and preprocess the image
    transform = get_cifar10_transforms()
    
    try:
        image = Image.open(image_path).convert('RGB')
        # Resize to 32x32 (CIFAR-10 size)
        image = image.resize((32, 32))
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        class_name = class_names[predicted_class]
        
        return predicted_class, confidence_score, class_name
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None


def test_on_cifar10_testset(model, num_samples=10, device="cuda"):
    """
    Test the model on CIFAR-10 test set.
    
    Args:
        model: The trained model
        num_samples: Number of test samples to evaluate
        device: Device to use
        
    Returns:
        accuracy: Test accuracy
        predictions: List of predictions
    """
    print(f"Testing on CIFAR-10 test set ({num_samples} samples)...")
    
    # Load CIFAR-10 test dataset
    transform = get_cifar10_transforms()
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=True
    )
    
    correct = 0
    total = 0
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.append({
                'true_label': labels.item(),
                'predicted_label': predicted.item(),
                'correct': predicted.item() == labels.item()
            })
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy, predictions


def visualize_predictions(model, num_samples=16, device="cuda"):
    """
    Visualize model predictions on random test samples.
    
    Args:
        model: The trained model
        num_samples: Number of samples to visualize
        device: Device to use
    """
    print(f"Visualizing {num_samples} predictions...")
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=True
    )
    
    # Get normalization transform for model input
    model_transform = get_cifar10_transforms()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Original image for display
            img_display = images[0].permute(1, 2, 0).numpy()
            
            # Preprocessed image for model (images are already tensors)
            img_model = images.to(device)
            # Apply normalization manually
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
            std = torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1).to(device)
            img_model = (img_model - mean) / std
            
            # Make prediction
            outputs = model(img_model)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Display image
            axes[i].imshow(img_display)
            axes[i].set_title(
                f"True: {class_names[labels.item()]}\n"
                f"Pred: {class_names[predicted.item()]}\n"
                f"Conf: {confidence.item():.2f}",
                fontsize=8
            )
            axes[i].axis('off')
            
            # Color code based on correctness
            if predicted.item() == labels.item():
                axes[i].set_facecolor('lightgreen')
            else:
                axes[i].set_facecolor('lightcoral')
    
    plt.tight_layout()
    plt.savefig('vgg11_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Predictions saved to vgg11_predictions.png")


def get_model_info(model):
    """
    Get information about the model architecture.
    
    Args:
        model: The VGG11 model
        
    Returns:
        info: Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': 'VGG11 with BatchNorm',
        'input_size': '3x32x32',
        'num_classes': 10,
        'dataset': 'CIFAR-10'
    }
    
    return info


def main():
    """Main function to demonstrate VGG11 usage."""
    print("VGG11 CIFAR-10 Pre-trained Model Demo")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load the pre-trained model
        model = load_pretrained_vgg11(device)
        
        # Get model information
        model_info = get_model_info(model)
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Test on CIFAR-10 test set
        print("\n" + "="*50)
        accuracy, predictions = test_on_cifar10_testset(model, num_samples=100, device=device)
        
        # Visualize some predictions
        print("\n" + "="*50)
        visualize_predictions(model, num_samples=16, device=device)
        
        # Example with a custom image (if available)
        print("\n" + "="*50)
        print("To test with your own image, use:")
        print("python use_vgg11_pretrained.py --image path/to/your/image.jpg")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Use pre-trained VGG11 on CIFAR-10")
    parser.add_argument("--image", type=str, help="Path to image file for prediction")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.image:
        # Load model and predict on single image
        device = args.device if torch.cuda.is_available() else "cpu"
        model = load_pretrained_vgg11(device)
        
        predicted_class, confidence, class_name = predict_single_image(
            model, args.image, device
        )
        
        if predicted_class is not None:
            print(f"\nPrediction Results:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Class Index: {predicted_class}")
    else:
        main() 