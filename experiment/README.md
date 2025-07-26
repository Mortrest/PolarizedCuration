# 🧪 CIFAR-10 Self-Consuming Retraining Experiment

This experiment simulates a self-consuming retraining loop where a generative model is trained on its own synthetic outputs, filtered via a reward function derived from a classifier, to study how different reward criteria affect alignment, bias, and collapse.

## 🎯 Research Goal

Study the effects of different reward functions on generative model behavior through iterative self-training:

1. **Mode Collapse**: Test if single-class rewards lead to complete mode collapse
2. **Bias Amplification**: Test if confidence-based rewards amplify classifier biases
3. **Alignment Dynamics**: Understand how reward functions affect model alignment over iterations

## 🧩 Experiment Components

### Core Components
- **Dataset**: CIFAR-10 (50,000 training images)
- **Generative Model**: Normalizing Flow with Optimal Transport Conditional Flow Matching (OT-CFM)
- **Classifier**: Pre-trained VGG11 (92.39% accuracy on CIFAR-10)
- **Curation**: Discrete K-choice filtering (Bradley-Terry model)
- **Retraining**: MLE on filtered synthetic samples

### Reward Functions
1. **Single Class Reward**: `r(x) = γ · q₀(x)` (targeting airplane class)
2. **Confidence Reward**: `r(x) = γ · max_i qᵢ(x)` (rewarding high confidence)

## 🔁 Experimental Procedure

### Step 1: Initial Training
- Train OT-CFM model on CIFAR-10 training data
- Save as initial distribution p₀

### Step 2: Iterative Retraining Loop (T iterations)
For each iteration t:

1. **Generate Synthetic Samples**: Sample 50,000 images from pₜ
2. **Compute Rewards**: Apply reward function using VGG11 classifier
3. **K-Choice Filtering**: 
   - Group samples into batches of K=4
   - Select samples using softmax over rewards: `P(select xₖ) = exp(r(xₖ)) / sum_j exp(r(xⱼ))`
   - Curate 2,500 samples total
4. **Retrain Model**: Fine-tune on curated samples (optionally mixed with real data)
5. **Track Metrics**: Class distribution, reward statistics, training loss

## 📊 Expected Outcomes

### Experiment 1: Single-Class Reward (Airplanes)
- **Prediction**: Model converges to only generate airplanes
- **Metrics**: Class diversity collapses, reward increases monotonically
- **Purpose**: Confirm mode collapse behavior

### Experiment 2: Confidence Reward
- **Prediction**: Reward increases, class imbalance emerges
- **Metrics**: Certain classes dominate due to classifier bias
- **Purpose**: Confirm bias amplification

## 🚀 Quick Start

### 1. Install Dependencies

**Option A: Full installation (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Minimal installation (for RTX 2080 or similar)**
```bash
pip install -r requirements_minimal.txt
```



### 2. Run Full Experiment
```bash
python experiment_runner.py
```

## 📁 File Structure

```
experiment/
├── config.py                 # Experiment configuration
├── generative_model.py       # OT-CFM model implementation
├── classifier.py             # VGG11 classifier wrapper
├── curation.py              # K-choice curation mechanism
├── data_loader.py           # CIFAR-10 data loading utilities
├── visualization.py         # Plotting and visualization tools
├── experiment_runner.py     # Main experiment orchestrator

├── requirements.txt         # Complete Python dependencies
├── requirements_minimal.txt # Minimal dependencies (RTX 2080 optimized)
├── README.md               # This file
├── data/                   # CIFAR-10 dataset (auto-downloaded)
├── models/                 # Saved model checkpoints
├── results/                # Experiment results and plots
└── logs/                   # Training logs
```

## ⚙️ Configuration

Key parameters in `config.py`:

```python
# Retraining Configuration
RETRAINING_CONFIG = {
    "num_iterations": 10,                    # Number of retraining loops
    "synthetic_samples_per_iteration": 50000, # Samples to generate
    "curated_samples_per_iteration": 2500,   # Samples to curate
    "real_samples_mix": 2500,                # Real samples to mix in
    "k_choice": 4,                          # K for K-choice filtering
    "reward_gamma": 5.0,                    # Reward scaling factor
    "retrain_steps": 50000,                 # Training steps per iteration
}

# Reward Functions
REWARD_CONFIGS = {
    "single_class": {
        "target_class": 0,  # Airplane class
        "gamma": 5.0,
    },
    "confidence": {
        "gamma": 5.0,
    }
}
```

## 📈 Monitoring and Visualization

The experiment automatically generates:

1. **Class Distribution Plots**: Track class diversity over iterations
2. **Reward Progression**: Monitor reward statistics
3. **Sample Grids**: Visualize generated samples at each iteration
4. **Training Loss Curves**: Track model convergence
5. **Experiment Summary**: Comprehensive overview plots

## 🔬 Key Metrics Tracked

### Per Iteration
- **Class Distribution**: Count of samples per class
- **Reward Statistics**: Mean, std, min, max of rewards
- **Curation Efficiency**: Ratio of curated to original samples
- **Training Loss**: Model convergence metrics

### Overall Trends
- **Mode Collapse**: Entropy of class distribution
- **Bias Amplification**: Class imbalance progression
- **Reward Optimization**: Reward improvement over iterations

## 🎛️ Customization

### Adding New Reward Functions
```python
# In classifier.py, add to get_rewards method:
elif reward_type == "your_reward":
    # Your reward computation
    rewards = your_reward_function(probabilities)
```

### Modifying Curation Strategy
```python
# In curation.py, modify _select_sample method:
def _select_sample(self, rewards):
    # Your selection strategy
    return selected_idx
```

### Changing Model Architecture
```python
# In generative_model.py, modify OTFlowModel class:
class YourFlowModel(OTFlowModel):
    def __init__(self, ...):
        # Your custom architecture
```

## 📊 Results Interpretation

### Mode Collapse Detection
- **Class Entropy**: Should decrease to near zero
- **Single Class Dominance**: One class should approach 100%
- **Reward Saturation**: Rewards should plateau at maximum

### Bias Amplification Detection
- **Class Imbalance**: Uneven distribution favoring certain classes
- **Confidence Correlation**: High confidence samples dominate
- **Classifier Bias**: Reflects inherent VGG11 biases

## 🔧 Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch sizes or sample counts
2. **Training Instability**: Increase real sample mixing ratio
3. **Slow Generation**: Use fewer integration steps or Euler method
4. **Component Errors**: Run `test_experiment.py` to isolate issues

### Performance Optimization
- Use GPU acceleration (CUDA)
- Reduce integration steps for faster sampling
- Use smaller model for faster training
- Enable mixed precision training

## 📚 References

- **Conditional Flow Matching**: [Tong et al., 2023](https://arxiv.org/abs/2302.00482)
- **Optimal Transport**: [Lipman et al., 2022](https://arxiv.org/abs/2209.15571)
- **Self-Consuming Models**: [Shumailov et al., 2023](https://arxiv.org/abs/2307.01850)
- **Bradley-Terry Model**: [Bradley & Terry, 1952](https://www.jstor.org/stable/2334029)

## 🤝 Contributing

To extend this experiment:

1. **New Reward Functions**: Add to `classifier.py`
2. **Alternative Curation**: Modify `curation.py`
3. **Different Models**: Extend `generative_model.py`
4. **Additional Metrics**: Enhance `visualization.py`

## 📄 License

This experiment is for research purposes. Please cite the original papers if using this code in your research.

---

**Note**: This experiment can be computationally intensive. Consider running on a GPU with sufficient memory and monitoring system resources during execution. 