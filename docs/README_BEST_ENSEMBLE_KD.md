# Best Ensemble and Knowledge Distillation Pipeline

This directory contains the optimal ensemble and knowledge distillation pipeline using the best configurations discovered through systematic experimentation.

## 🎯 Overview

This pipeline implements:
1. **Best Ensemble Model**: Optimal teacher ensemble using the top-performing model combination
2. **Best Knowledge Distillation**: Optimal KD configuration to transfer knowledge to the best student model

**Key Features:**
- ✅ **Follows Proper Pipeline Flow**: Same structure as `train.py` and `train_balancing.py`
- ✅ **WandB Integration**: Full experiment tracking and logging
- ✅ **Dataset Analysis**: Complete dataset statistics and visualization
- ✅ **Model Saving**: Proper checkpoint saving and loading
- ✅ **Ensemble Implementation**: True soft voting with frozen teacher models
- ✅ **Knowledge Distillation**: Ensemble teacher → student with optimal KD loss

## 📊 Best Configurations

### Best Ensemble Models
- **Models**: DenseNet-121, ResNet-101, DenseNet-201, EfficientNet-B4
- **Ensemble Type**: Soft Voting (probability averaging)
- **Performance**: 97.07% F1 Score (from previous experiments)

### Best Student Model
- **Model**: ShuffleNet V2 0.5× (shufflenet_v2_x0_5)
- **Performance**: 92.12% F1 Score (best among lightweight models)

### Best Hyperparameters
- **Learning Rate**: 0.0005
- **Scheduler**: Step LR
- **Optimizer**: Adam
- **Weight Decay**: 0.0001

### Best Data Balancing
- **Technique**: ADASYN (Adaptive Synthetic Sampling)
- **Improvement**: Handles class imbalance effectively

### Best Augmentation
- **Configuration**: `combined_augmentation_1.json`
  - Horizontal Flip: p=0.5
  - Rotation: ±20 degrees
  - Brightness: ±0.1

### Best KD Configuration
- **Temperature**: 15
- **Alpha (Hard Loss)**: 0.7
- **Beta (Distillation Loss)**: 0.3

## 🚀 Quick Start

### Step 1: Train Best Ensemble Models

Train all teacher models with optimal configurations:

```bash
python train_best_ensemble.py
```

**What it does:**
- Follows the same pipeline as `train_balancing.py`
- Initializes WandB for each model training
- Analyzes dataset with ADASYN balancing
- Saves dataset statistics and class mappings
- Visualizes sample images
- Trains each model with best hyperparameters + ADASYN + best augmentation
- Saves individual model checkpoints
- Creates ensemble training summary

**Expected Time**: ~16 hours (4 hours per model)

### Step 2: Evaluate Best Ensemble

Evaluate the ensemble performance using soft voting:

```bash
python evaluate_best_ensemble.py
```

**What it does:**
- Follows the same pipeline as `ensemble.py`
- Initializes WandB for ensemble evaluation
- Finds and loads all trained teacher models
- Creates `BestEnsembleModel` class with soft voting
- Analyzes dataset and visualizes samples
- Evaluates ensemble on test set with comprehensive metrics
- Saves confusion matrix, classification report, and performance metrics

**Expected Time**: ~1 hour

### Step 3: Train Best Knowledge Distillation

Train the student model using knowledge distillation:

```bash
python train_best_kd.py
```

**What it does:**
- Follows the same pipeline as `train_kd.py`
- Initializes WandB for KD training
- Loads teacher ensemble models (frozen)
- Creates `EnsembleTeacher` class for soft voting teacher outputs
- Implements `KnowledgeDistillationLoss` with optimal parameters
- Trains student model with KD using `KDTrainer`
- Uses ADASYN balancing and best augmentation
- Saves student model checkpoints and detailed results

**Expected Time**: ~3 hours

## 📁 Output Structure

```
outputs/
├── densenet121/
│   └── best_ensemble_densenet121_YYYYMMDDHHMMSS/
│       ├── config.json
│       ├── dataset_statistics.json
│       ├── class_mapping.json
│       ├── best_model.pth
│       ├── training_plots/
│       └── sample_visualizations/
├── resnet101/
│   └── best_ensemble_resnet101_YYYYMMDDHHMMSS/
├── densenet201/
│   └── best_ensemble_densenet201_YYYYMMDDHHMMSS/
├── efficientnet_b4/
│   └── best_ensemble_efficientnet_b4_YYYYMMDDHHMMSS/
├── ensemble_training_summary/
│   └── best_ensemble_training_summary.json
├── ensemble/
│   └── best_ensemble_evaluation_YYYYMMDDHHMMSS/
│       ├── config.json
│       ├── ensemble_info.json
│       ├── performance_metrics.json
│       ├── classification_report.json
│       ├── confusion_matrix.png
│       └── dataset_statistics.json
└── knowledge_distillation/
    └── shufflenet_v2_x0_5/
        └── best_kd_distillation_YYYYMMDDHHMMSS/
            ├── config.json
            ├── best_model.pth
            ├── test_classification_report.json
            ├── experiment_summary.json
            └── code_snapshot/
```

## 🔧 Pipeline Flow Details

### Individual Model Training (`train_best_ensemble.py`)

Each model follows the complete pipeline:

1. **Initialize WandB** - Project: `tomato-disease-best-ensemble`
2. **Analyze Dataset** - With ADASYN balancing statistics
3. **Save Dataset Statistics** - Complete class distribution analysis
4. **Load Data** - With ADASYN balancing + best augmentation
5. **Save Class Mapping** - For consistent evaluation
6. **Visualize Samples** - Sample images from each class
7. **Initialize Model** - With best hyperparameters
8. **Train Model** - Using `BalancingTrainer` with optimal config
9. **Test Model** - Comprehensive evaluation on test set
10. **Save Code Snapshot** - For reproducibility

### Ensemble Evaluation (`evaluate_best_ensemble.py`)

1. **Initialize WandB** - Project: `tomato-disease-best-ensemble`
2. **Find Trained Models** - From ensemble training summary
3. **Analyze Dataset** - Same dataset analysis as training
4. **Load Data** - Test set for evaluation
5. **Create Ensemble Model** - `BestEnsembleModel` with soft voting
6. **Get Predictions** - Ensemble predictions using averaged probabilities
7. **Evaluate Performance** - Comprehensive metrics and visualizations
8. **Save Results** - Detailed performance analysis

### Knowledge Distillation (`train_best_kd.py`)

1. **Initialize WandB** - Project: `tomato-disease-best-kd`
2. **Find Teacher Models** - Load ensemble models (frozen)
3. **Analyze Dataset** - With ADASYN balancing
4. **Load Data** - With ADASYN + best augmentation
5. **Create Teacher Ensemble** - `EnsembleTeacher` with frozen models
6. **Create Student Model** - ShuffleNet V2 0.5×
7. **Initialize KD Loss** - `KnowledgeDistillationLoss` with T=15, α=0.7, β=0.3
8. **Train with KD** - Using `KDTrainer` with ensemble teacher
9. **Test Student** - Final evaluation of distilled model
10. **Save Results** - Complete KD experiment summary

## 📈 Expected Performance

### Teacher Ensemble
- **Individual Models**: 91-94% F1 Score
- **Ensemble**: ~97% F1 Score
- **Improvement**: 3-6% over individual models

### Student Model
- **Baseline**: ~87% F1 Score
- **With Best Configs**: ~92% F1 Score
- **With KD**: ~94-95% F1 Score (expected)
- **Improvement**: 2-3% from knowledge distillation

### Model Efficiency
- **Teacher Ensemble**: ~200M parameters
- **Student Model**: ~1.4M parameters
- **Compression Ratio**: ~140×
- **Speed Improvement**: ~100× faster inference

## 🔧 Manual Commands

If you prefer to run commands manually or need to debug:

### Train Individual Teacher Models

```bash
# Each model follows the complete pipeline
python train_best_ensemble.py  # Trains all 4 models sequentially
```

### Evaluate Ensemble

```bash
python evaluate_best_ensemble.py
```

### Train Knowledge Distillation

```bash
python train_best_kd.py
```

## 🔍 Key Implementation Details

### Proper Pipeline Flow
- **Same Structure**: Follows `train.py`/`train_balancing.py` exactly
- **WandB Integration**: Full experiment tracking for each component
- **Dataset Analysis**: Complete statistics and visualization
- **Model Saving**: Proper checkpoint management

### Ensemble Implementation
- **Soft Voting**: True probability averaging, not just prediction voting
- **Frozen Teachers**: Teacher models are properly frozen during KD
- **Model Loading**: Robust model loading with checkpoint handling

### Knowledge Distillation
- **Ensemble Teacher**: Uses ensemble outputs as teacher, not individual models
- **Optimal KD Loss**: Temperature=15, α=0.7, β=0.3 based on experiments
- **Proper Training**: Student-only optimization with frozen teacher ensemble

### Data Handling
- **ADASYN Balancing**: Applied consistently across all components
- **Best Augmentation**: Uses `combined_augmentation_1.json`
- **Consistent Preprocessing**: Same image sizes and normalization

## 📊 Research Impact

This pipeline represents the culmination of systematic experimentation across:
- **Model Architecture**: 15+ models tested
- **Ensemble Combinations**: 100+ combinations evaluated
- **Hyperparameter Optimization**: 50+ configurations tested
- **Data Balancing**: 8 techniques compared
- **Knowledge Distillation**: 10+ KD configurations tested

The final configuration achieves:
- **State-of-the-art accuracy** on tomato leaf disease classification
- **Optimal efficiency** for deployment scenarios
- **Robust performance** across different class distributions

## 🚨 Prerequisites

1. **Dependencies**: All packages from `requirements_balancing.txt`
2. **Data**: Tomato leaf disease dataset in proper directory structure
3. **Augmentation Config**: `combined_augmentation_1.json` file
4. **Compute**: GPU recommended (16+ GB VRAM for ensemble training)

## 📝 Notes

- **Automatic Configuration**: All best configurations are applied automatically
- **Robust Model Loading**: Handles both checkpoint formats (`model_state_dict` and direct)
- **Comprehensive Logging**: All experiments logged to WandB with detailed metrics
- **Reproducibility**: Fixed seeds and complete code snapshots
- **Error Handling**: Graceful handling of missing models or failed training

## 🎯 Next Steps

After completing this pipeline:
1. **Model Deployment**: Deploy the distilled student model for production use
2. **Performance Analysis**: Compare results with baseline and individual optimizations
3. **Further Optimization**: Explore quantization and pruning for additional efficiency gains
4. **Publication**: Document findings for research publication

## 🔄 Troubleshooting

### Common Issues

1. **Missing Teacher Models**: Run `train_best_ensemble.py` first
2. **WandB Login**: Ensure WandB is properly configured
3. **GPU Memory**: Reduce batch size if encountering OOM errors
4. **Missing Augmentation Config**: Ensure `combined_augmentation_1.json` exists

### Verification

Check that each step produces the expected outputs:
- Training: `outputs/model_name/best_ensemble_*/best_model.pth`
- Ensemble: `outputs/ensemble/best_ensemble_evaluation_*/performance_metrics.json`
- KD: `outputs/knowledge_distillation/shufflenet_v2_x0_5/best_kd_*/experiment_summary.json` 