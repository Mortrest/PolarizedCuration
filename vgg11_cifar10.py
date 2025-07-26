#!/usr/bin/env python3
"""
VGG11 Classifier for CIFAR-10

This script provides a complete implementation for training and using a VGG11 classifier
on the CIFAR-10 dataset, based on the PyTorch-CIFAR10 repository:
https://github.com/jerett/PyTorch-CIFAR10/tree/master

The VGG11 model achieves ~91.25% accuracy on CIFAR-10 test set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
import json


class VGG11(nn.Module):
    """
    VGG11 architecture adapted for CIFAR-10 (32x32 images)
    
    Based on the implementation from:
    https://github.com/jerett/PyTorch-CIFAR10/blob/master/cifar10/classifiers/vgg.py
    """
    
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        
        # VGG11 configuration: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 2x2 -> 1x1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_cifar10_data_loaders(batch_size=64, train_split=0.8):
    """
    Get CIFAR-10 data loaders with train/validation split.
    
    Args:
        batch_size: Batch size for data loaders
        train_split: Fraction of training data to use for training (rest for validation)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Split training data into train and validation
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=150, device='cuda'):
    """
    Train the VGG11 model.
    
    Args:
        model: VGG11 model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        Training history and best model state
    """
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Training VGG11 for {num_epochs} epochs on {device}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate averages
        train_loss_avg = train_loss / len(train_loader)
        train_acc_avg = 100. * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc_avg = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(val_loss_avg)
        
        # Save best model
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            best_model_state = model.state_dict().copy()
        
        # Update history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc_avg)
        history['learning_rate'].append(current_lr)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.2f}%')
        print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 60)
    
    return history, best_model_state


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Test accuracy and predictions
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * correct / total
    test_loss_avg = test_loss / len(test_loader)
    
    print(f'Test Loss: {test_loss_avg:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return test_acc, all_predictions, all_targets


def save_model(model, model_state, history, save_path='vgg11_cifar10.pth'):
    """
    Save the trained model and training history.
    
    Args:
        model: Model architecture
        model_state: Best model state dict
        history: Training history
        save_path: Path to save the model
    """
    # Save model
    torch.save({
        'model_state_dict': model_state,
        'model_architecture': model,
        'history': history,
        'model_info': {
            'name': 'VGG11',
            'dataset': 'CIFAR-10',
            'num_classes': 10,
            'input_size': (3, 32, 32)
        }
    }, save_path)
    
    # Save history as JSON
    history_path = save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Model saved to {save_path}")
    print(f"Training history saved to {history_path}")


def load_model(model_path='vgg11_cifar10.pth', device='cuda'):
    """
    Load a trained VGG11 model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
    
    Returns:
        Loaded model and history
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model architecture
    model = checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load history
    history = checkpoint.get('history', {})
    
    print(f"Model loaded from {model_path}")
    print(f"Model info: {checkpoint.get('model_info', 'N/A')}")
    
    return model, history


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate
    ax3.plot(history['learning_rate'])
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)
    
    # Accuracy difference
    acc_diff = [val - train for val, train in zip(history['val_acc'], history['train_acc'])]
    ax4.plot(acc_diff)
    ax4.set_title('Validation - Training Accuracy Difference')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def predict_single_image(model, image_path, device='cuda'):
    """
    Predict class for a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image
        device: Device to run inference on
    
    Returns:
        Predicted class and confidence
    """
    from PIL import Image
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence


def main():
    """Main function to train or use VGG11 model."""
    parser = argparse.ArgumentParser(description='VGG11 CIFAR-10 Classifier')
    parser.add_argument('--mode', choices=['train', 'test', 'predict'], default='train',
                       help='Mode: train, test, or predict')
    parser.add_argument('--model_path', default='vgg11_cifar10.pth',
                       help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--image_path', default=None,
                       help='Path to image for prediction (required for predict mode)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Train the model
        print("Training VGG11 model on CIFAR-10...")
        
        # Get data loaders
        train_loader, val_loader, test_loader = get_cifar10_data_loaders(
            batch_size=args.batch_size
        )
        
        # Create model
        model = VGG11(num_classes=10)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        history, best_model_state = train_model(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, device=device
        )
        
        # Evaluate on test set
        model.load_state_dict(best_model_state)
        test_acc, _, _ = evaluate_model(model, test_loader, device=device)
        
        # Save model
        save_model(model, best_model_state, history, args.model_path)
        
        # Plot training history
        plot_training_history(history, 'vgg11_training_history.png')
        
    elif args.mode == 'test':
        # Test the model
        print("Testing VGG11 model...")
        
        # Load model
        model, history = load_model(args.model_path, device=device)
        
        # Get test data loader
        _, _, test_loader = get_cifar10_data_loaders(batch_size=args.batch_size)
        
        # Evaluate
        test_acc, predictions, targets = evaluate_model(model, test_loader, device=device)
        
        # Plot training history if available
        if history:
            plot_training_history(history, 'vgg11_training_history.png')
    
    elif args.mode == 'predict':
        # Predict on single image
        if not args.image_path:
            print("Error: --image_path is required for predict mode")
            return
        
        print(f"Predicting class for image: {args.image_path}")
        
        # Load model
        model, _ = load_model(args.model_path, device=device)
        
        # Predict
        predicted_class, confidence = predict_single_image(
            model, args.image_path, device=device
        )
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == '__main__':
    main() 