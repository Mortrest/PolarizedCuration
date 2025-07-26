"""
Main experiment runner for the self-consuming retraining loop.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any
import time
from tqdm import tqdm

# Import our modules
from config import ExperimentConfig
from generative_model import OTFlowModel, FlowTrainer
from classifier import VGG11Classifier
from curation import KChoiceCurator, RewardComputer
from data_loader import CIFAR10DataLoader, create_mixed_dataloader
from visualization import ExperimentVisualizer


class SelfConsumingExperiment:
    """
    Main experiment class for self-consuming retraining loop.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.DEVICE
        
        # Initialize components
        self.generative_model = None
        self.classifier = None
        self.curator = None
        self.reward_computer = None
        self.data_loader = None
        self.visualizer = None
        
        # Experiment tracking
        self.experiment_results = {
            'iterations': [],
            'class_distributions': [],
            'reward_stats': [],
            'training_losses': [],
            'reward_type': None
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all experiment components."""
        print("Initializing experiment components...")
        
        # Initialize data loader
        self.data_loader = CIFAR10DataLoader(
            data_dir=self.config.PATHS['data_dir'],
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            device=self.device
        )
        
        # Initialize classifier
        self.classifier = VGG11Classifier(device=self.device)
        
        # Initialize curator
        self.curator = KChoiceCurator(
            k=self.config.RETRAINING_CONFIG['k_choice'],
            device=self.device
        )
        
        # Initialize reward computer
        self.reward_computer = RewardComputer(self.classifier, device=self.device)
        
        # Initialize visualizer
        self.visualizer = ExperimentVisualizer(save_dir=self.config.PATHS['results_dir'])
        
        print("All components initialized successfully!")
    
    def load_or_train_initial_model(self) -> OTFlowModel:
        """
        Load pre-trained model or train initial model on CIFAR-10.
        
        Returns:
            Initial generative model
        """
        print("Setting up initial generative model...")
        
        # Check if pre-trained model exists
        pretrained_path = self.config.PATHS['pretrained_flow']
        if os.path.exists(pretrained_path):
            print(f"Loading pre-trained model from {pretrained_path}")
            try:
                self.generative_model = OTFlowModel.load_model(pretrained_path, device=self.device)
                print("Pre-trained model loaded successfully!")
            except Exception as e:
                print(f"Failed to load pre-trained model: {e}")
                print("Training initial model from scratch...")
                self._create_and_train_initial_model()
        else:
            print(f"Pre-trained model not found at {pretrained_path}")
            print("Training initial model from scratch...")
            self._create_and_train_initial_model()
        
        return self.generative_model
    
    def _create_and_train_initial_model(self):
        """Create and train the initial model from scratch."""
        self.generative_model = OTFlowModel(
            image_size=self.config.GENERATIVE_MODEL_CONFIG['image_size'],
            in_channels=self.config.GENERATIVE_MODEL_CONFIG['in_channels'],
            hidden_size=self.config.GENERATIVE_MODEL_CONFIG['hidden_size'],
            num_blocks=self.config.GENERATIVE_MODEL_CONFIG['num_blocks'],
            num_heads=self.config.GENERATIVE_MODEL_CONFIG['num_heads'],
            dropout=self.config.GENERATIVE_MODEL_CONFIG['dropout'],
            device=self.device
        )
        
        # Train the initial model
        self._train_initial_model()
    
    def _train_initial_model(self):
        """Train the initial generative model on CIFAR-10."""
        print("Training initial generative model...")
        
        # Create trainer
        trainer = FlowTrainer(
            model=self.generative_model,
            learning_rate=self.config.GENERATIVE_MODEL_CONFIG['learning_rate'],
            warmup_steps=self.config.GENERATIVE_MODEL_CONFIG['warmup'],
            device=self.device
        )
        
        # Get training data
        train_loader = self.data_loader.get_train_loader()
        
        # Train the model
        losses = trainer.train_on_dataset(
            train_loader,
            num_steps=self.config.GENERATIVE_MODEL_CONFIG['total_steps'],
            log_interval=self.config.LOGGING_CONFIG['log_interval']
        )
        
        # Save the model
        model_path = os.path.join(self.config.PATHS['models_dir'], 'initial_model.pt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.generative_model.save_model(model_path)
        
        print(f"Initial model trained and saved to {model_path}")
    
    def generate_synthetic_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate synthetic samples using the current generative model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        print(f"Generating {num_samples} synthetic samples...")
        
        samples = []
        batch_size = 100  # Generate in batches to avoid memory issues
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
                current_batch_size = min(batch_size, num_samples - i)
                batch_samples = self.generative_model.sample(
                    batch_size=current_batch_size,
                    num_steps=50,  # Reduced for faster generation
                    integration_method="euler"
                )
                samples.append(batch_samples)
        
        all_samples = torch.cat(samples, dim=0)
        print(f"Generated {len(all_samples)} synthetic samples")
        
        return all_samples
    
    def run_single_iteration(
        self,
        iteration: int,
        reward_type: str,
        target_class: int = 0,
        gamma: float = 5.0
    ) -> Dict[str, Any]:
        """
        Run a single iteration of the retraining loop.
        
        Args:
            iteration: Current iteration number
            reward_type: Type of reward function
            target_class: Target class for single_class reward
            gamma: Scaling factor for rewards
            
        Returns:
            Dictionary with iteration results
        """
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration}")
        print(f"{'='*50}")
        
        # Step 1: Generate synthetic samples
        synthetic_samples = self.generate_synthetic_samples(
            self.config.RETRAINING_CONFIG['synthetic_samples_per_iteration']
        )
        
        # Step 2: Compute rewards
        print("Computing rewards...")
        rewards = self.reward_computer.compute_rewards(
            synthetic_samples, reward_type, target_class, gamma
        )
        
        # Step 3: Get class distribution
        class_distribution = self.reward_computer.get_class_distribution(synthetic_samples)
        class_distribution_np = class_distribution.cpu().numpy()
        
        # Step 4: Curate samples using K-choice
        curated_samples, curated_rewards = self.curator.curate_samples(
            synthetic_samples,
            rewards,
            self.config.RETRAINING_CONFIG['curated_samples_per_iteration']
        )
        
        # Step 5: Get curation statistics
        curation_stats = self.curator.get_curation_stats(
            synthetic_samples,
            rewards,
            self.config.RETRAINING_CONFIG['curated_samples_per_iteration']
        )
        
        # Step 6: Mix with real samples (optional)
        if self.config.RETRAINING_CONFIG['real_samples_mix'] > 0:
            print("Mixing with real samples...")
            real_samples = self.data_loader.get_train_subset(
                self.config.RETRAINING_CONFIG['real_samples_mix']
            )
            # Extract samples from dataloader
            real_samples_list = []
            for batch in real_samples:
                real_samples_list.append(batch[0])
                if len(torch.cat(real_samples_list, dim=0)) >= self.config.RETRAINING_CONFIG['real_samples_mix']:
                    break
            real_samples_tensor = torch.cat(real_samples_list, dim=0)[:self.config.RETRAINING_CONFIG['real_samples_mix']]
            # Move to CPU for DataLoader compatibility
            real_samples_tensor = real_samples_tensor.cpu()
            
            # Create mixed dataloader
            mixed_loader = create_mixed_dataloader(
                real_samples_tensor,
                curated_samples,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.NUM_WORKERS
            )
        else:
            # Use only curated samples
            from data_loader import SyntheticDataset
            dataset = SyntheticDataset(curated_samples)
            mixed_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                drop_last=True
            )
        
        # Step 7: Retrain the model
        print("Retraining generative model...")
        trainer = FlowTrainer(
            model=self.generative_model,
            learning_rate=self.config.GENERATIVE_MODEL_CONFIG['learning_rate'],
            warmup_steps=1000,  # Shorter warmup for retraining
            device=self.device
        )
        
        training_losses = trainer.train_on_dataset(
            mixed_loader,
            num_steps=self.config.RETRAINING_CONFIG['retrain_steps'],
            log_interval=self.config.LOGGING_CONFIG['log_interval']
        )
        
        # Step 8: Save model checkpoint
        checkpoint_path = os.path.join(
            self.config.PATHS['models_dir'],
            f'model_iteration_{iteration}.pt'
        )
        self.generative_model.save_model(checkpoint_path)
        
        # Step 9: Generate visualization samples
        if iteration % self.config.LOGGING_CONFIG['plot_interval'] == 0:
            sample_path = f'samples_iteration_{iteration}.png'
            self.visualizer.plot_sample_grid(
                curated_samples,
                num_samples=16,
                save_path=sample_path,
                title=f"Curated Samples - Iteration {iteration}"
            )
        
        # Compile results
        results = {
            'iteration': iteration,
            'class_distribution': class_distribution_np,
            'curation_stats': curation_stats,
            'training_losses': training_losses,
            'num_synthetic': len(synthetic_samples),
            'num_curated': len(curated_samples),
            'mean_reward': rewards.mean().item(),
            'mean_curated_reward': curated_rewards.mean().item() if len(curated_rewards) > 0 else 0.0
        }
        
        print(f"Iteration {iteration} completed!")
        print(f"Mean reward: {results['mean_reward']:.4f}")
        print(f"Mean curated reward: {results['mean_curated_reward']:.4f}")
        print(f"Class distribution: {class_distribution_np}")
        
        return results
    
    def run_experiment(
        self,
        reward_type: str = "confidence",
        target_class: int = 0,
        gamma: float = 5.0
    ):
        """
        Run the complete self-consuming retraining experiment.
        
        Args:
            reward_type: Type of reward function ("confidence" or "single_class")
            target_class: Target class for single_class reward
            gamma: Scaling factor for rewards
        """
        print(f"Starting Self-Consuming Retraining Experiment")
        print(f"Reward Type: {reward_type}")
        print(f"Target Class: {target_class} ({self.config.CLASS_NAMES[target_class]})")
        print(f"Gamma: {gamma}")
        print(f"Number of Iterations: {self.config.RETRAINING_CONFIG['num_iterations']}")
        
        # Set reward type in results
        self.experiment_results['reward_type'] = reward_type
        
        # Load or train initial model
        self.load_or_train_initial_model()
        
        # Run iterations
        for iteration in range(self.config.RETRAINING_CONFIG['num_iterations']):
            start_time = time.time()
            
            # Run single iteration
            results = self.run_single_iteration(
                iteration, reward_type, target_class, gamma
            )
            
            # Store results
            self.experiment_results['iterations'].append(iteration)
            self.experiment_results['class_distributions'].append(results['class_distribution'])
            self.experiment_results['reward_stats'].append(results['curation_stats'])
            self.experiment_results['training_losses'].append(results['training_losses'])
            
            # Generate plots
            if iteration % self.config.LOGGING_CONFIG['plot_interval'] == 0:
                self.visualizer.plot_class_distribution(
                    self.experiment_results['class_distributions'],
                    self.config.CLASS_NAMES,
                    self.experiment_results['iterations'],
                    f"class_distribution_{reward_type}.png"
                )
                
                self.visualizer.plot_reward_progression(
                    self.experiment_results['reward_stats'],
                    self.experiment_results['iterations'],
                    f"reward_progression_{reward_type}.png"
                )
            
            iteration_time = time.time() - start_time
            print(f"Iteration {iteration} took {iteration_time:.2f} seconds")
        
        # Generate final summary
        print("\nGenerating final summary...")
        self.visualizer.plot_experiment_summary(
            self.experiment_results,
            f"experiment_summary_{reward_type}.png"
        )
        
        # Save experiment data
        self.visualizer.save_experiment_data(
            self.experiment_results,
            f"experiment_results_{reward_type}.json"
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to {self.config.PATHS['results_dir']}")


def main():
    """Main function to run the experiment."""
    # Create configuration
    config = ExperimentConfig()
    
    # Create experiment
    experiment = SelfConsumingExperiment(config)
    
    # Run experiments
    print("Running experiments...")
    
    # Experiment 1: Single Class Reward (Airplane)
    print("\n" + "="*60)
    print("EXPERIMENT 1: Single Class Reward (Airplane)")
    print("="*60)
    experiment.run_experiment(
        reward_type="single_class",
        target_class=0,  # Airplane
        gamma=5.0
    )
    
    # Reset for next experiment
    experiment.experiment_results = {
        'iterations': [],
        'class_distributions': [],
        'reward_stats': [],
        'training_losses': [],
        'reward_type': None
    }
    
    # Experiment 2: Confidence Reward
    print("\n" + "="*60)
    print("EXPERIMENT 2: Confidence Reward")
    print("="*60)
    experiment.run_experiment(
        reward_type="confidence",
        gamma=5.0
    )
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main() 