"""
Visualization utilities for the retraining experiment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any
import os


class ExperimentVisualizer:
    """
    Visualization class for tracking experiment progress and results.
    """
    
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_class_distribution(
        self,
        class_distributions: List[np.ndarray],
        class_names: List[str],
        iterations: List[int],
        save_path: str = "class_distribution.png"
    ):
        """
        Plot class distribution over iterations.
        
        Args:
            class_distributions: List of class distributions for each iteration
            class_names: Names of the classes
            iterations: Iteration numbers
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to DataFrame for easier plotting
        data = []
        for i, dist in enumerate(class_distributions):
            for j, count in enumerate(dist):
                data.append({
                    'Iteration': iterations[i],
                    'Class': class_names[j],
                    'Count': count
                })
        
        df = pd.DataFrame(data)
        
        # Plot
        for i, class_name in enumerate(class_names):
            class_data = df[df['Class'] == class_name]
            ax.plot(class_data['Iteration'], class_data['Count'], 
                   marker='o', label=class_name, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution Over Iterations')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reward_progression(
        self,
        reward_stats: List[Dict[str, float]],
        iterations: List[int],
        save_path: str = "reward_progression.png"
    ):
        """
        Plot reward progression over iterations.
        
        Args:
            reward_stats: List of reward statistics for each iteration
            iterations: Iteration numbers
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics
        original_means = [stats['original_mean_reward'] for stats in reward_stats]
        curated_means = [stats['curated_mean_reward'] for stats in reward_stats]
        improvements = [stats['reward_improvement'] for stats in reward_stats]
        curation_ratios = [stats['curation_ratio'] for stats in reward_stats]
        
        # Plot original vs curated mean rewards
        ax1.plot(iterations, original_means, 'b-o', label='Original', linewidth=2)
        ax1.plot(iterations, curated_means, 'r-s', label='Curated', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Mean Reward Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot reward improvement
        ax2.plot(iterations, improvements, 'g-^', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Reward Improvement')
        ax2.set_title('Reward Improvement Over Iterations')
        ax2.grid(True, alpha=0.3)
        
        # Plot curation ratio
        ax3.plot(iterations, curation_ratios, 'm-d', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Curation Ratio')
        ax3.set_title('Curation Ratio Over Iterations')
        ax3.grid(True, alpha=0.3)
        
        # Plot reward distribution
        ax4.hist(original_means, bins=10, alpha=0.7, label='Original', color='blue')
        ax4.hist(curated_means, bins=10, alpha=0.7, label='Curated', color='red')
        ax4.set_xlabel('Mean Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Mean Rewards')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sample_grid(
        self,
        samples: torch.Tensor,
        num_samples: int = 16,
        save_path: str = "sample_grid.png",
        title: str = "Generated Samples"
    ):
        """
        Plot a grid of generated samples.
        
        Args:
            samples: Tensor of samples
            num_samples: Number of samples to display
            save_path: Path to save the plot
            title: Title for the plot
        """
        # Denormalize samples to [0, 1] range
        if samples.max() <= 1.0:
            # Already in [0, 1] range
            samples_display = samples
        else:
            # Denormalize from [-1, 1] to [0, 1]
            samples_display = (samples + 1) / 2
        
        # Select samples
        if len(samples) > num_samples:
            indices = torch.randperm(len(samples))[:num_samples]
            samples_display = samples_display[indices]
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(samples_display):
                    img = samples_display[idx].permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)
                    axes[i, j].imshow(img)
                axes[i, j].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_loss(
        self,
        losses: List[List[float]],
        iterations: List[int],
        save_path: str = "training_loss.png"
    ):
        """
        Plot training loss over iterations.
        
        Args:
            losses: List of loss curves for each iteration
            iterations: Iteration numbers
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, loss_curve in enumerate(losses):
            steps = np.arange(len(loss_curve))
            ax.plot(steps, loss_curve, alpha=0.7, label=f'Iteration {iterations[i]}')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_experiment_summary(
        self,
        experiment_results: Dict[str, Any],
        save_path: str = "experiment_summary.png"
    ):
        """
        Create a comprehensive summary plot of the experiment.
        
        Args:
            experiment_results: Dictionary containing all experiment results
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        iterations = experiment_results.get('iterations', [])
        class_distributions = experiment_results.get('class_distributions', [])
        reward_stats = experiment_results.get('reward_stats', [])
        reward_type = experiment_results.get('reward_type', 'Unknown')
        
        # Plot 1: Class distribution heatmap
        if class_distributions:
            class_data = np.array(class_distributions)
            im1 = ax1.imshow(class_data.T, cmap='viridis', aspect='auto')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Class')
            ax1.set_title('Class Distribution Heatmap')
            ax1.set_yticks(range(10))
            ax1.set_yticklabels(['airplane', 'auto', 'bird', 'cat', 'deer', 
                                'dog', 'frog', 'horse', 'ship', 'truck'])
            plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Reward progression
        if reward_stats:
            original_means = [stats['original_mean_reward'] for stats in reward_stats]
            curated_means = [stats['curated_mean_reward'] for stats in reward_stats]
            
            ax2.plot(iterations, original_means, 'b-o', label='Original', linewidth=2)
            ax2.plot(iterations, curated_means, 'r-s', label='Curated', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Mean Reward')
            ax2.set_title(f'Reward Progression ({reward_type})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Class diversity (entropy)
        if class_distributions:
            entropies = []
            for dist in class_distributions:
                # Compute entropy of class distribution
                probs = dist / dist.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            ax3.plot(iterations, entropies, 'g-^', linewidth=2)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Class Diversity (Entropy)')
            ax3.set_title('Class Diversity Over Iterations')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Reward improvement
        if reward_stats:
            improvements = [stats['reward_improvement'] for stats in reward_stats]
            ax4.plot(iterations, improvements, 'm-d', linewidth=2)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Reward Improvement')
            ax4.set_title('Reward Improvement Over Iterations')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Experiment Summary: {reward_type}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_experiment_data(
        self,
        experiment_results: Dict[str, Any],
        save_path: str = "experiment_results.json"
    ):
        """
        Save experiment results to JSON file.
        
        Args:
            experiment_results: Dictionary containing all experiment results
            save_path: Path to save the data
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in experiment_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_results[key] = [v.tolist() for v in value]
            else:
                serializable_results[key] = value
        
        with open(os.path.join(self.save_dir, save_path), 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Experiment data saved to {os.path.join(self.save_dir, save_path)}") 