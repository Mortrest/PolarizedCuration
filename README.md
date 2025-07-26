# RebuttalEmbraceDiv: Self-Consuming Generative Model Experiment

This repository contains a comprehensive implementation of a self-consuming retraining experiment using Optimal Transport Conditional Flow Matching (OT-CFM) on CIFAR-10, with VGG11 classifier-based reward functions.

## 🎯 Project Overview

The experiment studies how different reward criteria affect alignment, bias, and collapse in generative models through a self-consuming retraining loop. The generative model (OT-CFM) is trained on its own synthetic outputs, filtered via reward functions derived from a pretrained VGG11 classifier.

### Key Features

- **OT-CFM Generative Model**: Uses pre-trained Optimal Transport Conditional Flow Matching
- **VGG11 Classifier**: Pre-trained VGG11 for reward computation
- **K-Choice Curation**: Bradley-Terry model for sample filtering
- **Multiple Reward Functions**: Single Class Reward and Confidence Reward
- **Comprehensive Visualization**: Class distribution, reward progression, and sample grids
- **Scaled for RTX 2080**: Optimized parameters for mid-range GPUs

## 📁 Repository Structure

```
RebuttalEmbraceDiv/
├── experiment/                    # Main experiment directory
│   ├── config.py                 # Experiment configuration
│   ├── generative_model.py       # OT-CFM model implementation
│   ├── classifier.py             # VGG11 classifier wrapper
│   ├── curation.py               # K-choice curation mechanism
│   ├── data_loader.py            # CIFAR-10 data loading utilities
│   ├── visualization.py          # Plotting and visualization
│   ├── experiment_runner.py      # Main experiment orchestration
│   ├── requirements.txt          # Full dependencies
│   ├── requirements_minimal.txt  # Minimal dependencies
│   └── README.md                 # Experiment documentation
├── use_pretrained_model.py       # OT-CFM model usage examples
├── train_flow_model.py           # Training script for flow models
├── example_usage.py              # Various usage examples
├── use_vgg11_pretrained.py       # VGG11 classifier usage
├── simple_vgg11_example.py       # Simple VGG11 example
├── vgg11_cifar10.py              # VGG11 CIFAR-10 implementation
├── README_VGG11.md               # VGG11 documentation
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd RebuttalEmbraceDiv

# Install dependencies
pip install -r experiment/requirements_minimal.txt
```

### 2. Download Pre-trained Weights

The experiment requires pre-trained weights for both OT-CFM and VGG11:

- **OT-CFM**: `pretrained_weights/otcfm_cifar10_weights_step_400000.pt`
- **VGG11**: `PyTorch_CIFAR10/cifar10_models/state_dicts/vgg11_bn.pt`

### 3. Run the Experiment

```bash
cd experiment
python experiment_runner.py
```

This will run both experiments:
- **Single Class Reward (Airplane)**: Rewards based on airplane class probability
- **Confidence Reward**: Rewards based on maximum class confidence

## 🔬 Experiment Details

### Reward Functions

1. **Single Class Reward**: 
   - Rewards samples based on probability of a specific class (default: airplane)
   - Formula: `R(x) = γ * P(class=target | x)`

2. **Confidence Reward**:
   - Rewards samples based on maximum class confidence
   - Formula: `R(x) = γ * max(P(class | x))`

### K-Choice Curation

Uses a Bradley-Terry like model (softmax over rewards) to filter samples:
- Groups K samples together
- Selects one sample using softmax probabilities over rewards
- Helps maintain diversity while improving quality

### Self-Consuming Loop

1. Generate synthetic samples using current model
2. Compute rewards using VGG11 classifier
3. Curate samples using K-choice filtering
4. Mix with real samples for stability
5. Retrain model on curated data
6. Repeat for specified iterations


