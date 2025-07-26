"""
Configuration file for the CIFAR-10 Retraining Experiment.
"""

import torch

class ExperimentConfig:
    """Configuration class for the self-consuming retraining experiment."""
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset configuration (RTX 2080 optimized)
    DATASET_SIZE = 50000  # Number of training images
    BATCH_SIZE = 64  # Reduced from 128
    NUM_WORKERS = 2  # Reduced from 4
    
    # Generative model configuration (RTX 2080 optimized)
    GENERATIVE_MODEL_CONFIG = {
        "image_size": 32,
        "in_channels": 3,
        "hidden_size": 64,  # Reduced from 128
        "num_blocks": 2,  # Reduced from 4
        "num_heads": 4,  # Reduced from 8
        "dropout": 0.1,
        "learning_rate": 2e-4,
        "total_steps": 20000,  # Reduced for faster experimentation
        "warmup": 2000,
        "ema_decay": 0.9999,
        "grad_clip": 1.0
    }
    
    # Retraining configuration (RTX 2080 optimized)
    RETRAINING_CONFIG = {
        "num_iterations": 5,  # Reduced from 10
        "synthetic_samples_per_iteration": 1000,  # Reduced from 50000
        "curated_samples_per_iteration": 500,  # Reduced from 2500
        "real_samples_mix": 500,  # Reduced from 2500
        "k_choice": 4,  # K for K-choice filtering
        "reward_gamma": 5.0,  # γ parameter for reward scaling
        "retrain_steps": 1000,  # Reduced from 50000
    }
    
    # Reward functions configuration
    REWARD_CONFIGS = {
        "single_class": {
            "name": "Single Class Reward (Airplane)",
            "target_class": 0,  # Airplane class
            "gamma": 5.0,
            "description": "Reward based on airplane class probability"
        },
        "confidence": {
            "name": "Confidence Reward",
            "gamma": 5.0,
            "description": "Reward based on maximum class confidence"
        }
    }
    
    # Paths
    PATHS = {
        "data_dir": "./data",
        "models_dir": "./models",
        "results_dir": "./results",
        "logs_dir": "./logs",
        "pretrained_flow": "../pretrained_weights/otcfm_cifar10_weights_step_400000.pt",
        "pretrained_classifier": "../PyTorch_CIFAR10/cifar10_models/state_dicts/vgg11_bn.pt"
    }
    
    # CIFAR-10 class names
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Logging configuration
    LOGGING_CONFIG = {
        "log_interval": 1000,
        "save_interval": 10000,
        "eval_interval": 5000,
        "plot_interval": 1  # Plot every iteration
    } 