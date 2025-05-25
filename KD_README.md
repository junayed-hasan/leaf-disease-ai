# Knowledge Distillation Implementation

This implementation provides a comprehensive knowledge distillation pipeline for tomato leaf disease classification, following the systematic research plan outlined in your project.

## 🏗️ Architecture Overview

### Core Components

1. **`kd_model.py`** - Knowledge distillation model wrapper
   - Combines teacher ensemble with student model
   - Implements logit and feature-level distillation
   - Supports basic and advanced temperature scaling
   - L2 distance for feature-level KD

2. **`trainer_kd.py`** - KD-specific trainer
   - Mirrors the structure of your existing `trainer.py`
   - Tracks multiple loss components (CE, KD, Feature)
   - Comprehensive logging and evaluation

3. **`train_kd.py`** - Main KD training script
   - Follows the same flow as your existing `train.py`
   - Configurable hyperparameters via command line
   - Automatic teacher/student model discovery

4. **`run_kd_experiments.py`** - Experiment automation
   - Systematic execution of your 4-step research plan
   - Progress tracking and error handling
   - 110 total experiments as planned

## 🚀 Quick Start

### Single Experiment

```bash
# Basic KD with temperature=4
python train_kd.py --student_model shufflenet_v2 --temperature 4

# Advanced temperature scaling
python train_kd.py --student_model shufflenet_v2 --teacher_temp 6 --student_temp 3

# Feature-only distillation
python train_kd.py --student_model shufflenet_v2 --teacher_temp 5 --student_temp 1 --use_feature_distillation --beta 0 --alpha 0.5 --gamma 0.5

# Combined distillation
python train_kd.py --student_model shufflenet_v2 --teacher_temp 4 --student_temp 1 --use_feature_distillation
```

### Systematic Experiments

```bash
# Run all experiments (30 total)
python run_kd_experiments.py

# Run specific steps
python run_kd_experiments.py --step 1  # Basic KD (10 experiments)
python run_kd_experiments.py --step 2  # Feature-only distillation (10 experiments)
python run_kd_experiments.py --step 3  # Combined distillation (10 experiments)
```

## 📊 Research Plan Implementation

### Step 1: Basic KD Screening (10 experiments)
- **Model**: shufflenet_v2 (best performing student)
- **Temperatures**: [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
- **Goal**: Identify optimal temperature for shufflenet_v2

### Step 2: Feature-Only Distillation (10 experiments)
- **Model**: shufflenet_v2
- **Configuration**: α=0.5 (CE), β=0 (no logit KD), γ=0.5 (feature KD)
- **Temperatures**: [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
- **Goal**: Evaluate feature-level distillation effectiveness

### Step 3: Combined Distillation (10 experiments)
- **Model**: shufflenet_v2
- **Configuration**: α=β=γ=1/3 (equal weights)
- **Temperatures**: [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
- **Goal**: Best of both logit and feature distillation

**Total: 30 experiments** (reduced from 110 for faster execution)

## 🔧 Configuration Options

### Basic Parameters
- `--student_model`: Student architecture (required)
- `--teacher_models`: Teacher ensemble (default: best ensemble from your results)
- `--dataset`: Dataset to use (default: combined)

### KD Loss Weights
- `--alpha`: Classification loss weight (default: 1/3)
- `--beta`: Logit distillation loss weight (default: 1/3)
- `--gamma`: Feature distillation loss weight (default: 1/3)

### Temperature Scaling
- `--temperature`: Basic KD temperature (default: 4.0)
- `--teacher_temp`: Teacher temperature for advanced scaling
- `--student_temp`: Student temperature for advanced scaling

### Features
- `--use_feature_distillation`: Enable feature-level KD
- `--experiment_name`: Custom experiment name

## 📁 Output Structure

```
outputs/knowledge_distillation/
├── shufflenet_v2/
│   ├── kd_densenet121_resnet101_densenet201_efficientnet_b4_to_shufflenet_v2_T4_20241201123456/
│   │   ├── best_model.pth              # Complete KD model
│   │   ├── best_student_model.pth      # Student model only
│   │   ├── config.json                 # Experiment configuration
│   │   ├── test_metrics.csv            # Detailed metrics
│   │   ├── confusion_matrix.png        # Confusion matrix
│   │   └── code_snapshot/              # Code backup
│   └── ...
└── mobilenet_v2/
    └── ...
```

## 🎯 Key Features

### Teacher Ensemble Integration
- Automatically loads the best ensemble: densenet121 + resnet101 + densenet201 + efficientnet_b4
- Soft voting with temperature scaling
- Frozen teacher parameters during training

### Feature Extraction
- Penultimate layer features (global averaged 1D vectors)
- Automatic dimension matching with adaptation layers
- L2 distance loss for feature similarity

### Robust Training
- Starts from pretrained student weights when available
- Early stopping and learning rate scheduling
- Comprehensive logging with wandb integration
- Automatic teacher model discovery

### Error Handling
- Graceful handling of missing teacher/student models
- Timeout protection (2 hours per experiment)
- Detailed error reporting

## 📈 Expected Results

Based on your baseline results:
- **Baseline ShuffleNetV2 Performance**: 92.12% F1-score
- **Teacher Ensemble Performance**: 97.07% F1-score
- **Expected KD Improvement**: 2-4% F1-score improvement over student baseline

## 🔍 Monitoring Progress

### Real-time Monitoring
- Rich console output with progress bars
- Live loss tracking (CE, KD, Feature components)
- WandB integration for experiment tracking

### Results Analysis
After experiments, analyze:
1. Which temperature values work best for shufflenet_v2
2. Feature vs. logit distillation effectiveness
3. Optimal temperature for combined approach

## 🚨 Important Notes

1. **Teacher Model Requirements**: Ensure all teacher models (densenet121, resnet101, densenet201, efficientnet_b4) have trained weights in the `outputs/` directory

2. **GPU Memory**: Feature distillation requires additional GPU memory. Monitor usage during combined experiments.

3. **Experiment Duration**: Each experiment takes ~30-60 minutes. Total suite should complete in ~15-30 hours.

4. **Result Tracking**: All experiments are logged to wandb project "tomato-disease-kd"

## 🔄 Next Steps

After completing the experiments:
1. Analyze results to identify best configurations
2. Run hyperparameter optimization on loss weights (α, β, γ)
3. Consider ensemble of distilled students
4. Compare with original teacher ensemble efficiency

## 📞 Usage Support

For specific use cases:

```bash
# Test single model quickly
python train_kd.py --student_model custom_cnn_1m --temperature 3 --experiment_name test_run

# Debug feature extraction
python train_kd.py --student_model shufflenet_v2 --use_feature_distillation --gamma 1.0 --alpha 0 --beta 0

# Custom teacher ensemble
python train_kd.py --student_model mobilenet_v2 --teacher_models densenet121 resnet101 --temperature 5
```

The implementation is designed to be modular, robust, and true to your existing codebase structure. Happy experimenting! 🎉 