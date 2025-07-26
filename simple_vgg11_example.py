#!/usr/bin/env python3
"""
Simple example of using the pre-trained VGG11 model on CIFAR-10.

This script demonstrates the basic usage of the VGG11 model for image classification.
"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Add the PyTorch_CIFAR10 directory to the path
sys.path.append('./PyTorch_CIFAR10')

from cifar10_models.vgg import vgg11_bn


def main():
    """Simple example of VGG11 usage."""
    print("VGG11 CIFAR-10 Classification Example")
    print("=" * 40)
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pre-trained model
    print("Loading VGG11 model...")
    model = vgg11_bn(pretrained=True, device="cpu")
    model = model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616]
        )
    ])
    
    # Example: Create a simple test image (you can replace this with your own image)
    print("\nCreating a test image...")
    
    # Create a simple colored square as test image
    from torchvision.utils import save_image
    test_image = torch.randn(3, 32, 32)  # Random image
    save_image(test_image, 'test_image.png')
    
    # Load and preprocess the test image
    image = Image.open('test_image.png').convert('RGB')
    image = image.resize((32, 32))  # Resize to CIFAR-10 size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get results
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    predicted_class_name = class_names[predicted_class]
    
    # Display results
    print(f"\nPrediction Results:")
    print(f"  Predicted Class: {predicted_class_name}")
    print(f"  Class Index: {predicted_class}")
    print(f"  Confidence: {confidence_score:.4f}")
    
    # Show all class probabilities
    print(f"\nAll Class Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  {class_names[i]:12}: {prob.item():.4f}")
    
    print(f"\nTest image saved as 'test_image.png'")
    print("You can replace this with your own image for testing.")


if __name__ == "__main__":
    main() 