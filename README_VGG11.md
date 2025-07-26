# VGG11 Pre-trained Model on CIFAR-10

This repository provides a pre-trained VGG11 model for CIFAR-10 classification, based on the [PyTorch_CIFAR10 repository](https://github.com/huyvnphan/PyTorch_CIFAR10) by huyvnphan.

## Overview

The VGG11 model has been trained on the CIFAR-10 dataset and achieves **92.39% validation accuracy**. The model includes batch normalization and is optimized for 32x32 RGB images.

## Model Specifications

- **Architecture**: VGG11 with BatchNorm
- **Input Size**: 3×32×32 (RGB images)
- **Output**: 10 classes (CIFAR-10 categories)
- **Parameters**: 28.15M
- **Model Size**: ~107 MB
- **Accuracy**: 92.39% on validation set

## CIFAR-10 Classes

The model can classify images into these 10 categories:
1. airplane
2. automobile  
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

2. Install dependencies:
```bash
pip install -r requirements_vgg11.txt
```

3. Download pre-trained weights:
```bash
cd PyTorch_CIFAR10
python train.py --download_weights 1
cd ..
```

## Quick Start

### Basic Usage

```python
from use_vgg11_pretrained import load_pretrained_vgg11, predict_single_image

# Load the model
model = load_pretrained_vgg11(device="cuda")

# Predict on an image
predicted_class, confidence, class_name = predict_single_image(
    model, "path/to/your/image.jpg", device="cuda"
)

print(f"Predicted: {class_name} (confidence: {confidence:.2f})")
```

### Command Line Usage

Run the demo script:
```bash
python use_vgg11_pretrained.py
```

This will:
- Load the pre-trained VGG11 model
- Test it on 100 CIFAR-10 test samples
- Generate a visualization of predictions
- Show model information

Predict on a single image:
```bash
python use_vgg11_pretrained.py --image path/to/your/image.jpg
```

## Data Preprocessing

The model expects input images to be:
- **Size**: 32×32 pixels (will be resized automatically)
- **Format**: RGB
- **Normalization**: 
  - Mean: [0.4914, 0.4822, 0.4465]
  - Std: [0.2471, 0.2435, 0.2616]

## Performance

The pre-trained model achieves:
- **Test Accuracy**: ~95% on random test samples
- **Inference Speed**: Fast on GPU
- **Memory Usage**: ~107 MB for model weights

## Model Architecture

The VGG11 model consists of:
- **Convolutional layers**: 8 conv layers with batch normalization
- **Max pooling**: 5 pooling layers
- **Fully connected**: 3 FC layers (4096 → 4096 → 10)
- **Activation**: ReLU
- **Dropout**: Applied in FC layers

## Available Models

The repository also includes pre-trained weights for other models:
- VGG13_bn (94.22% accuracy)
- VGG16_bn (94.00% accuracy)  
- VGG19_bn (93.95% accuracy)
- ResNet18 (93.07% accuracy)
- ResNet34 (93.34% accuracy)
- ResNet50 (93.65% accuracy)
- DenseNet121 (94.06% accuracy)
- MobileNetV2 (93.91% accuracy)
- And more...

## Training Details

The model was trained with:
- **Dataset**: CIFAR-10 (50,000 training images)
- **Optimizer**: SGD with momentum
- **Learning Rate**: 1e-2 with weight decay
- **Batch Size**: 256
- **Epochs**: 100
- **Data Augmentation**: Random horizontal flip

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{huyvnphan2020pytorch,
  title={PyTorch models trained on CIFAR-10 dataset},
  author={huyvnphan},
  year={2020},
  publisher={GitHub},
  url={https://github.com/huyvnphan/PyTorch_CIFAR10}
}
```

## License

This code is based on the PyTorch_CIFAR10 repository and follows its MIT license terms.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA (recommended for inference)
- 4GB+ GPU memory (for training)

## File Structure

```
.
├── PyTorch_CIFAR10/           # Original repository with weights
├── use_vgg11_pretrained.py    # Main usage script
├── requirements_vgg11.txt     # Dependencies
├── README_VGG11.md           # This file
├── vgg11_predictions.png     # Generated visualization
└── data/                     # CIFAR-10 dataset (downloaded automatically)
``` 