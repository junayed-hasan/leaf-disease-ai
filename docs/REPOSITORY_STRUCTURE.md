# Repository Structure

This document provides a comprehensive overview of the TomatoLeaf-AI repository structure after reorganization.

## 📁 Directory Overview

```
TomatoLeaf-AI/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 setup.py                     # Package installation script
├── 📄 requirements.txt             # Core dependencies
├── 📄 requirements_balancing.txt   # Data balancing dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 🖼️ archi_tomato.png             # Architecture diagram
├── 📄 access.tex                   # Research paper LaTeX source
│
├── 📂 src/                         # Source code (modular organization)
│   ├── 📂 configurations/          # Configuration files
│   ├── 📂 data_augmentation/       # Data augmentation strategies
│   ├── 📂 data_balancing/          # ADASYN-based balancing
│   ├── 📂 datasets/                # Dataset loaders and utilities
│   ├── 📂 distillation/            # Knowledge distillation
│   ├── 📂 ensemble/                # Ensemble learning
│   ├── 📂 evaluation/              # Model evaluation scripts
│   ├── 📂 explainable_ai/          # Interpretability analysis
│   ├── 📂 hyperparameter_tuning/   # Hyperparameter optimization
│   ├── 📂 models/                  # Model architectures
│   ├── 📂 quantization/            # Model quantization
│   └── 📂 utils/                   # Utility functions
│
├── 📂 data/                        # Datasets (organized by source)
│   ├── 📂 combined/                # Unified cross-domain dataset
│   ├── 📂 plantvillage/           # PlantVillage dataset (lab conditions)
│   ├── 📂 tomatovillage/          # TomatoVillage dataset (field conditions)
│   └── 📂 [other datasets]/       # Additional evaluation datasets
│
├── 📂 docs/                        # Comprehensive documentation
│   ├── 📄 PROJECT_OVERVIEW.md      # Detailed project overview
│   ├── 📄 INSTALLATION.md          # Installation guide
│   ├── 📄 REPOSITORY_STRUCTURE.md  # This file
│   └── 📄 [other docs]/            # Additional documentation
│
├── 📂 scripts/                     # Training and utility scripts
│   └── 📄 train.py                 # Main training script
│
├── 📂 mobile_app/                  # Flutter mobile application
│   ├── 📂 lib/                     # Flutter source code
│   ├── 📂 android/                 # Android-specific files
│   ├── 📂 ios/                     # iOS-specific files
│   └── 📄 pubspec.yaml             # Flutter dependencies
│
├── 📂 outputs/                     # Model outputs and results
│   ├── 📂 models/                  # Trained model checkpoints
│   ├── 📂 logs/                    # Training logs
│   └── 📂 visualizations/          # Result visualizations
│
└── 📂 checkpoints/                 # Model checkpoints
    ├── 📂 ensemble/                # Ensemble model checkpoints
    ├── 📂 distilled/               # Knowledge distilled models
    └── 📂 quantized/               # Quantized models
```

## 🔧 Source Code Organization

### Core Modules

#### `src/configurations/`
Configuration management for all experiments:
- `config.py` - Main configuration file
- `hyperparameter_config.py` - Hyperparameter settings
- `data_balancing_config.py` - Data balancing parameters
- `augmentation_config.py` - Data augmentation settings

#### `src/models/`
Model architectures and factory patterns:
- **CNN Models**: ResNet, DenseNet, EfficientNet families
- **Lightweight Models**: MobileNet, ShuffleNet, SqueezeNet
- **Vision Transformers**: ViT, Swin Transformer variants
- **Custom Models**: Lightweight CNN architectures
- `model_factory.py` - Unified model creation interface

#### `src/datasets/`
Dataset handling and preprocessing:
- `dataset.py` - Main dataset classes
- `plantdoc_dataset.py` - PlantDoc dataset loader
- Data loading utilities and transformations

### Specialized Modules

#### `src/ensemble/`
Ensemble learning implementation:
- `ensemble.py` - Ensemble model class
- `train_best_ensemble.py` - Ensemble training script
- `evaluate_best_ensemble.py` - Ensemble evaluation

#### `src/distillation/`
Knowledge distillation framework:
- `kd_model.py` - Knowledge distillation model
- `train_kd.py` - KD training script
- `trainer_kd.py` - KD trainer class
- `train_best_kd.py` - Optimized KD training
- `run_kd_experiments.py` - Systematic KD experiments

#### `src/quantization/`
Model quantization and optimization:
- `mobile_quantization_pipeline.py` - Mobile quantization
- `computational_analysis.py` - Performance analysis
- `onnx_model_evaluator.py` - ONNX model evaluation
- Quantized model evaluation scripts

#### `src/explainable_ai/`
Interpretability and explainability:
- `interpretability_analysis.py` - Grad-CAM++, LIME analysis
- Attention visualization utilities
- Biological validation tools

### Experimental Modules

#### `src/data_augmentation/`
Data augmentation strategies:
- `train_augmentation.py` - Augmentation training
- `run_augmentation_experiments.py` - Systematic experiments
- Augmentation configuration files

#### `src/data_balancing/`
Class imbalance handling:
- `dataset_balancing.py` - ADASYN implementation
- `train_balancing.py` - Balanced training
- `trainer_balancing.py` - Balancing trainer
- `run_balancing_experiments.py` - Balancing experiments

#### `src/hyperparameter_tuning/`
Hyperparameter optimization:
- `train_hyperparameter.py` - HP tuning training
- `trainer_hp.py` - HP tuning trainer
- `run_hyperparameter_experiments.py` - Systematic HP search

#### `src/evaluation/`
Comprehensive evaluation framework:
- Cross-domain evaluation scripts
- Performance benchmarking tools
- Statistical analysis utilities
- Visualization generators

## 📊 Data Organization

### Dataset Structure
```
data/
├── combined/                       # Unified dataset (15 classes)
│   ├── train/
│   │   ├── Healthy/
│   │   ├── Bacterial_Spot/
│   │   ├── Early_Blight/
│   │   └── [other classes]/
│   ├── val/
│   └── test/
│
├── plantvillage/                   # Laboratory conditions
│   └── color/
│       ├── Tomato___healthy/
│       ├── Tomato___Bacterial_spot/
│       └── [other classes]/
│
└── tomatovillage/                  # Field conditions
    └── Variant-a(Multiclass Classification)/
        ├── train/
        ├── val/
        └── test/
```

### Class Mapping
The repository includes comprehensive class harmonization:
- **15 Unified Classes**: Standardized across all datasets
- **Cross-Domain Compatibility**: Consistent labeling scheme
- **Balanced Distribution**: ADASYN-based balancing

## 📱 Mobile Application

### Flutter App Structure
```
mobile_app/
├── lib/
│   ├── main.dart                   # App entry point
│   ├── models/                     # Data models
│   ├── screens/                    # UI screens
│   ├── services/                   # ML inference services
│   └── utils/                      # Utility functions
├── android/                        # Android configuration
├── ios/                           # iOS configuration
└── assets/                        # App assets (models, images)
```

### Features
- **Real-time Inference**: Camera-based disease detection
- **Offline Capability**: On-device model execution
- **Multilingual Support**: English, Bengali, Spanish
- **Treatment Recommendations**: Disease-specific advice

## 📚 Documentation Structure

### Comprehensive Documentation
```
docs/
├── PROJECT_OVERVIEW.md             # Technical methodology
├── INSTALLATION.md                 # Setup instructions
├── REPOSITORY_STRUCTURE.md         # This file
├── API_REFERENCE.md                # API documentation
├── TRAINING_GUIDE.md               # Training instructions
├── DEPLOYMENT_GUIDE.md             # Deployment guide
└── [specialized docs]/             # Module-specific docs
```

### Documentation Categories
1. **Setup & Installation**: Getting started guides
2. **Technical Overview**: Research methodology
3. **API Reference**: Code documentation
4. **Tutorials**: Step-by-step guides
5. **Deployment**: Production deployment

## 🔄 Workflow Integration

### Development Workflow
1. **Configuration**: Set parameters in `src/configurations/`
2. **Data Preparation**: Use `src/datasets/` for data loading
3. **Model Training**: Execute scripts in respective modules
4. **Evaluation**: Use `src/evaluation/` for assessment
5. **Deployment**: Quantize and deploy via `src/quantization/`

### Experiment Tracking
- **Weights & Biases**: Integrated experiment logging
- **Configuration Management**: YAML-based parameter tracking
- **Reproducibility**: Fixed seeds and version control

## 🎯 Key Benefits of This Structure

### Modularity
- **Separation of Concerns**: Each module handles specific functionality
- **Reusability**: Components can be used independently
- **Maintainability**: Easy to update and extend

### Scalability
- **Easy Extension**: Add new models, datasets, or methods
- **Parallel Development**: Multiple developers can work simultaneously
- **Version Control**: Clean git history with organized changes

### Research Reproducibility
- **Systematic Organization**: Clear experimental structure
- **Configuration Management**: Centralized parameter control
- **Documentation**: Comprehensive guides and references

### Production Readiness
- **Mobile Integration**: Flutter app with quantized models
- **Deployment Scripts**: Automated deployment pipeline
- **Performance Monitoring**: Computational analysis tools

---

This structure provides a solid foundation for both research and production deployment, ensuring maintainability, scalability, and reproducibility of the TomatoLeaf-AI framework. 