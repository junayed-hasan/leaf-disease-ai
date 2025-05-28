# Data Balancing Experiments for Tomato Leaf Disease Classification

This module implements comprehensive data balancing techniques to improve the performance of tomato leaf disease classification models, particularly for handling class imbalance issues.

## Overview

The data balancing system implements 5 major techniques:
1. **Random Oversampling** - Randomly duplicate minority class samples
2. **SMOTE** - Synthetic Minority Oversampling Technique
3. **ADASYN** - Adaptive Synthetic Sampling
4. **Offline Augmentation** - Pre-generate augmented samples for balancing
5. **Focal Loss** - Loss function that focuses on hard examples

## Key Features

- **Best Hyperparameters Integration** - Uses optimized hyperparameters (lr=0.0005, scheduler=step, optimizer=adam, weight_decay=0.0001)
- **Best Augmentation Configuration** - Automatically loads best augmentation settings
- **Modular Design** - Each technique can be run independently
- **Comprehensive Analysis** - Detailed per-class metrics, confidence analysis, and imbalance-specific evaluation
- **Systematic Experiments** - Automated running of all techniques with proper comparison
- **Research-Ready Output** - Organized results suitable for research paper analysis

## Installation

Install additional dependencies for data balancing:

```bash
pip install -r requirements_balancing.txt
```

Key new dependencies:
- `imbalanced-learn` - For SMOTE, ADASYN, and random oversampling
- `opencv-python` - For offline augmentation processing

## File Structure

```
├── data_balancing_config.py          # Configuration for all balancing techniques
├── dataset_balancing.py              # Dataset loading with balancing applied
├── trainer_balancing.py              # Trainer with balancing-specific functionality
├── train_balancing.py                # Main training script for balancing experiments
├── run_balancing_experiments.py      # Systematic experiment runner
├── requirements_balancing.txt        # Additional dependencies
└── README_balancing.md              # This file
```

## Usage

### 1. Single Experiment

Run a single balancing technique:

```bash
# Baseline (no balancing)
python train_balancing.py --model shufflenet_v2 --dataset combined

# Random Oversampling
python train_balancing.py --model shufflenet_v2 --dataset combined --balancing_technique random_oversampling

# SMOTE
python train_balancing.py --model shufflenet_v2 --dataset combined --balancing_technique smote

# ADASYN  
python train_balancing.py --model shufflenet_v2 --dataset combined --balancing_technique adasyn

# Offline Augmentation
python train_balancing.py --model shufflenet_v2 --dataset combined --balancing_technique offline_augmentation

# Focal Loss only
python train_balancing.py --model shufflenet_v2 --dataset combined --loss_function focal_loss

# Combined: SMOTE + Focal Loss
python train_balancing.py --model shufflenet_v2 --dataset combined --balancing_technique smote --loss_function focal_loss
```

### 2. Systematic Experiments

Run all balancing experiments systematically:

```bash
# Run all experiment groups
python run_balancing_experiments.py --model shufflenet_v2 --dataset combined

# Run specific group only
python run_balancing_experiments.py --model shufflenet_v2 --dataset combined --group resampling
python run_balancing_experiments.py --model shufflenet_v2 --dataset combined --group loss_function
python run_balancing_experiments.py --model shufflenet_v2 --dataset combined --group combined
```

### 3. Available Groups

- **baseline** - No balancing (1 experiment)
- **resampling** - Random oversampling, SMOTE, ADASYN (3 experiments)
- **augmentation** - Offline augmentation (1 experiment)  
- **loss_function** - Focal loss only (1 experiment)
- **combined** - Combined techniques (2 experiments)

**Total: 8 experiments**

## Data Balancing Techniques

### 1. Random Oversampling
- **Method**: Randomly duplicates samples from minority classes
- **Pros**: Simple, preserves original data distribution
- **Cons**: Can lead to overfitting, increases dataset size significantly

### 2. SMOTE (Synthetic Minority Oversampling Technique)
- **Method**: Creates synthetic samples by interpolating between existing minority samples
- **Pros**: Reduces overfitting compared to random oversampling
- **Cons**: May create unrealistic samples, computationally expensive

### 3. ADASYN (Adaptive Synthetic Sampling)
- **Method**: Similar to SMOTE but focuses on harder-to-learn minority samples
- **Pros**: Better adaptation to data distribution
- **Cons**: More complex, may be unstable for very small classes

### 4. Offline Augmentation
- **Method**: Pre-generates augmented samples using best augmentation techniques
- **Pros**: Uses domain-appropriate transformations, controllable
- **Cons**: Increases storage requirements, limited by augmentation quality

### 5. Focal Loss
- **Method**: Modifies loss function to focus on hard examples and rare classes
- **Pros**: No data modification needed, addresses loss imbalance directly
- **Cons**: Requires hyperparameter tuning (alpha, gamma)

## Output Structure

Each experiment creates organized outputs:

```
outputs/
└── {model}_balance_{technique}_{loss}/
    ├── config.json                    # Experiment configuration
    ├── dataset_statistics.json        # Dataset analysis
    ├── class_mapping.json            # Class to index mapping
    ├── best_model.pth                 # Best trained model
    ├── test_metrics.csv               # Detailed test metrics
    ├── confusion_matrix.png           # Confusion matrix visualization
    ├── confidence_analysis.csv        # Prediction confidence analysis
    ├── confidence_distribution.png    # Confidence distribution plot
    ├── balancing_analysis.json        # Balancing-specific analysis
    ├── experiment_summary.json        # Final experiment summary
    └── code_snapshot/                 # Code used for experiment
```

## Key Analysis Metrics

### 1. Standard Metrics
- **Accuracy** - Overall classification accuracy
- **Precision, Recall, F1-Score** - Per-class and averaged metrics
- **Macro F1** - Unweighted average F1 (important for imbalanced data)
- **Weighted F1** - Sample-weighted average F1

### 2. Imbalance-Specific Metrics
- **Per-class Performance** - Individual class precision/recall/F1
- **Confidence Analysis** - Prediction confidence by correctness
- **Class Distribution Changes** - Before/after balancing comparison
- **Imbalance Ratio** - Ratio of majority to minority class

### 3. Research Analysis
- **Computational Overhead** - Training time comparison
- **Memory Usage** - Dataset size changes
- **Generalization** - Validation vs. test performance
- **Class-wise Improvement** - Which classes benefit most

## Best Practices

### 1. Experiment Design
- Always run baseline first for comparison
- Use macro F1-score as primary metric for imbalanced evaluation
- Monitor per-class metrics to understand technique effects
- Consider computational cost vs. performance trade-offs

### 2. Data Considerations
- Ensure validation/test sets remain unchanged (imbalance preserved)
- Only apply balancing to training set
- Consider domain-specific constraints for synthetic sample generation

### 3. Analysis Guidelines
- Compare techniques on macro F1, not just accuracy
- Analyze confidence distributions to understand model certainty
- Look for overfitting indicators (train vs. val performance)
- Consider practical deployment constraints

## Integration with Research Pipeline

This balancing system integrates with your existing research pipeline:

1. **Uses Best Hyperparameters** - Automatically applies optimal lr, optimizer, scheduler, weight_decay
2. **Uses Best Augmentations** - Loads `combined_augmentation_1.json` configuration
3. **Modular Design** - Can be combined with other techniques (KD, ensemble, etc.)
4. **Consistent Output Format** - Matches existing experiment structure
5. **Wandb Integration** - Automatic experiment tracking and comparison

## Expected Results

Based on research literature, expected improvements:

- **SMOTE/ADASYN**: 2-5% improvement in macro F1-score
- **Focal Loss**: 1-3% improvement, especially for rare classes  
- **Offline Augmentation**: 1-4% improvement with good augmentation strategy
- **Combined Techniques**: Potential 3-7% improvement over baseline

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use lighter balancing techniques
2. **SMOTE Failures**: Ensure sufficient samples per class (k_neighbors parameter)
3. **Slow Training**: Consider using subset of data for technique comparison
4. **Focal Loss Convergence**: Adjust gamma parameter (try 1.0, 2.0, 3.0)

### Performance Tips

1. **Feature Extraction for SMOTE**: Uses resized 32x32 images for efficiency
2. **Offline Augmentation**: Pre-computed augmentations avoid runtime overhead
3. **Focal Loss**: No data preprocessing overhead
4. **Batch Processing**: Optimized data loading for large datasets

## Research Applications

This system is designed for:

- **Ablation Studies** - Systematic comparison of balancing techniques
- **Method Development** - Testing new balancing approaches
- **Performance Analysis** - Understanding when techniques help/hurt
- **Publication Ready** - Comprehensive metrics and visualizations
- **Reproducible Research** - Fixed seeds, documented configurations

## Next Steps

After running balancing experiments:

1. **Combine with Knowledge Distillation** - Apply best balancing technique to KD experiments
2. **Ensemble Integration** - Use balanced models in ensemble approaches
3. **Cross-Dataset Validation** - Test best techniques across different datasets
4. **Deployment Optimization** - Consider inference-time constraints for final model selection 