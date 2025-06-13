# Knowledge Distillation Experiments

## Overview
This pipeline implements systematic knowledge distillation experiments for tomato leaf disease classification, focusing on finding optimal balance between classification loss (α) and distillation loss (β).

## Experiment Design

### Teacher Ensemble
- **Models**: DenseNet-121, ResNet-101, DenseNet-201, EfficientNet-B4
- **Performance**: 97.07% F1-score (best ensemble from previous experiments)
- **Dataset**: Combined (15 classes)

### Student Model
- **Architecture**: ShuffleNetV2
- **Baseline Performance**: 92.12% F1-score
- **Target**: 2-4% improvement through knowledge distillation

### Experiment Configuration

#### Step 1: Alpha/Beta Exploration
- **Objective**: Find optimal balance between classification loss (α) and distillation loss (β)
- **Constraint**: α + β = 1.0 (no feature distillation: γ = 0.0)
- **Alpha Values**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- **Beta Values**: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
- **Total Experiments**: 11

### Training Configuration
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Temperature**: 4.0 (standard for knowledge distillation)
- **Dataset**: Combined (15 classes)

## Usage

### Run All Experiments
```bash
python run_kd_experiments.py --step 1
```

### Individual Experiment
```bash
python train_kd.py \
    --student_model shufflenet_v2 \
    --alpha 0.7 \
    --beta 0.3 \
    --gamma 0.0 \
    --temperature 4.0 \
    --dataset combined \
    --epochs 50
```

## Expected Results

### Alpha/Beta Analysis
- **α = 1.0, β = 0.0**: Standard supervised training (baseline)
- **α = 0.0, β = 1.0**: Pure knowledge distillation
- **α = 0.5, β = 0.5**: Balanced approach
- **Optimal Range**: Expected around α = 0.6-0.8, β = 0.2-0.4

### Performance Target
- **Baseline**: 92.12% (ShuffleNetV2 without KD)
- **Target**: 94-96% (2-4% improvement)
- **Teacher Ensemble**: 97.07% (upper bound)

## File Structure

```
├── run_kd_experiments.py     # Main experiment runner
├── train_kd.py              # Individual KD training script
├── kd_model.py              # KD model implementation
├── trainer_kd.py            # KD trainer class
├── outputs/
│   └── knowledge_distillation/
│       └── shufflenet_v2/   # Experiment results
└── README_KD_EXPERIMENTS.md # This file
```

## Results Analysis

### Metrics to Monitor
1. **Student F1-Score**: Primary metric for comparison
2. **Classification Loss**: α component effectiveness
3. **KL Divergence Loss**: β component effectiveness
4. **Total Loss**: Combined optimization target
5. **Validation Accuracy**: Overfitting detection

### Best Configuration Selection
The optimal α/β combination will be selected based on:
1. Highest validation F1-score
2. Stable training convergence
3. Balanced loss components
4. Test set generalization

## Timeline
- **Total Experiments**: 11
- **Estimated Time**: ~22-33 hours (2-3 hours per experiment)
- **Completion**: Analysis and final model selection

## Next Steps
After completing the α/β exploration:
1. Analyze results and identify optimal configuration
2. Train final model with best parameters
3. Compare against baseline and teacher ensemble
4. Document findings and improvements 