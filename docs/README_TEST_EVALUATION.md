# Test Dataset Evaluation

This module provides scripts to evaluate your trained ensemble and knowledge distillation models on the test sets of your original datasets (tomatoVillage and plantVillage). This validates model performance on the original test data that the models were designed for.

## 🎯 Overview

The test dataset evaluation system:
1. **Evaluates on original test sets**: Tests models on tomatoVillage and plantVillage test datasets
2. **Comprehensive analysis**: Full performance metrics, confusion matrices, misclassification analysis
3. **Model comparison**: Direct comparison between ensemble and student model performance
4. **Efficiency analysis**: Model size vs performance trade-offs

## 📁 Files Overview

### Core Scripts

1. **`test_dataset_loader.py`** - Dataset loader for original test sets
2. **`evaluate_ensemble_on_test_datasets.py`** - Evaluates ensemble model on test sets
3. **`evaluate_kd_on_test_datasets.py`** - Evaluates KD student model on test sets

### Key Features

- ✅ **Automatic Test Set Detection**: Finds available test datasets automatically
- ✅ **WandB Integration**: Full experiment tracking and logging
- ✅ **Dataset Analysis**: Complete statistics and sample visualization  
- ✅ **Model Loading**: Robust loading of trained ensemble and KD models
- ✅ **Comprehensive Evaluation**: Detailed metrics, confusion matrices, misclassification analysis
- ✅ **Results Saving**: All outputs saved in organized directory structure
- ✅ **Performance Comparison**: Side-by-side comparison of both models

## 🚀 Quick Start

### Prerequisites

1. **Trained Models**: Both ensemble models and KD student model
2. **Test Datasets**: Either `tomatovillage/test/` or `plantvillage/test/` directories
3. **Dependencies**: All packages from your existing requirements files

### Step 1: Evaluate Ensemble Model on Test Sets

```bash
python evaluate_ensemble_on_test_datasets.py
```

**What it does:**
- Automatically detects available test datasets (tomatovillage, plantvillage)
- Loads your trained ensemble models
- Evaluates ensemble using soft voting on each test set
- Generates comprehensive performance analysis for each dataset
- Saves confusion matrices, classification reports, and misclassification analysis

**Expected Time**: ~20-40 minutes per dataset

### Step 2: Evaluate Student Model on Test Sets

```bash
python evaluate_kd_on_test_datasets.py
```

**What it does:**
- Automatically detects available test datasets
- Loads your trained KD student model
- Evaluates student model performance on each test set
- Compares efficiency (model size vs performance)
- Saves detailed performance analysis for each dataset

**Expected Time**: ~10-20 minutes per dataset

## 📁 Output Structure

```
outputs/
└── test_evaluation/
    ├── ensemble/
    │   └── ensemble_on_test_datasets_YYYYMMDDHHMMSS/
    │       ├── config.json
    │       ├── tomatovillage_test_dataset_statistics.json
    │       ├── tomatovillage_test_sample_visualizations.png
    │       ├── tomatovillage_ensemble_performance_metrics.json
    │       ├── tomatovillage_ensemble_classification_report.json
    │       ├── tomatovillage_ensemble_confusion_matrix.png
    │       ├── tomatovillage_ensemble_misclassifications.json
    │       ├── plantvillage_test_dataset_statistics.json
    │       ├── plantvillage_test_sample_visualizations.png
    │       ├── plantvillage_ensemble_performance_metrics.json
    │       ├── plantvillage_ensemble_classification_report.json
    │       ├── plantvillage_ensemble_confusion_matrix.png
    │       ├── plantvillage_ensemble_misclassifications.json
    │       └── test_evaluation_summary.json
    └── student/
        └── student_on_test_datasets_YYYYMMDDHHMMSS/
            ├── config.json
            ├── tomatovillage_test_dataset_statistics.json
            ├── tomatovillage_test_sample_visualizations.png
            ├── tomatovillage_student_performance_metrics.json
            ├── tomatovillage_student_classification_report.json
            ├── tomatovillage_student_confusion_matrix.png
            ├── tomatovillage_student_misclassifications.json
            ├── plantvillage_test_dataset_statistics.json
            ├── plantvillage_test_sample_visualizations.png
            ├── plantvillage_student_performance_metrics.json
            ├── plantvillage_student_classification_report.json
            ├── plantvillage_student_confusion_matrix.png
            ├── plantvillage_student_misclassifications.json
            └── test_evaluation_summary.json
```

## 🔧 Pipeline Flow Details

### Ensemble Evaluation (`evaluate_ensemble_on_test_datasets.py`)

1. **Detect Test Datasets** - Automatically find available test datasets
2. **Initialize WandB** - Project: `tomato-disease-test-evaluation`
3. **Load Class Mapping** - Original combined dataset class mapping
4. **Find Ensemble Models** - Automatically locate trained ensemble models
5. **Create Ensemble** - Load all models and create soft voting ensemble
6. **For Each Test Dataset:**
   - Analyze test dataset statistics
   - Visualize test samples
   - Load test data with DataLoader
   - Get ensemble predictions
   - Evaluate performance with comprehensive metrics
   - Save results and visualizations
7. **Save Overall Summary** - Combined results across all datasets

### Student Model Evaluation (`evaluate_kd_on_test_datasets.py`)

1. **Detect Test Datasets** - Automatically find available test datasets
2. **Find KD Model** - Locate trained student model from KD
3. **Initialize WandB** - Project: `tomato-disease-test-evaluation`
4. **Load Class Mapping** - Original combined dataset class mapping
5. **Create Student Evaluator** - Load student model
6. **For Each Test Dataset:**
   - Analyze test dataset statistics
   - Visualize test samples
   - Load test data with model-specific image size
   - Get student model predictions
   - Evaluate performance including efficiency analysis
   - Save results and visualizations
7. **Save Overall Summary** - Combined results across all datasets

## 📈 Expected Performance

### Performance Comparison

| Model Type | Parameters | Expected F1 Score | Speed | Use Case |
|---|---|---|---|---|
| **Ensemble** | ~200M | 95-99% | Slow (4 models) | Research/High Accuracy |
| **Student (KD)** | ~1.4M | 90-95% | Fast (1 model) | Production/Deployment |

### Dataset Comparison

| Dataset | Classes | Test Samples | Description |
|---|---|---|---|
| **tomatoVillage** | 10 | ~2,000-3,000 | Subset focused on key diseases |
| **plantVillage** | 15 | ~5,000-8,000 | Full dataset with all classes |

### Key Metrics Tracked

- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro)**: Unweighted average F1 across classes
- **F1 Score (Weighted)**: Sample-weighted average F1
- **Per-Class Metrics**: Precision, Recall, F1 for each class
- **Misclassification Analysis**: Detailed error analysis
- **Model Efficiency**: Parameters vs performance trade-off (for student model)

## 🔍 Key Features

### Automatic Dataset Discovery
- **Test Set Detection**: Automatically finds tomatovillage/test and plantvillage/test
- **Flexible Dataset Support**: Works with available datasets, skips missing ones
- **Class Mapping**: Uses original dataset class mappings automatically

### Robust Model Loading
- **Ensemble Models**: Automatically finds trained ensemble models from training summary
- **KD Student Model**: Finds your specific trained student model
- **Fallback Search**: Robust search mechanisms if summaries not found

### Comprehensive Analysis
- **Performance Metrics**: All standard classification metrics
- **Visual Analysis**: Confusion matrices and sample visualizations
- **Error Analysis**: Detailed misclassification analysis with confidence scores
- **Efficiency Analysis**: Model size vs performance trade-offs
- **Cross-Dataset Comparison**: Performance across different test sets

### Professional Output
- **Organized Results**: All outputs saved in timestamped directories
- **JSON Reports**: Machine-readable results for further analysis
- **WandB Logging**: Full experiment tracking and comparison
- **Visual Reports**: High-quality plots and visualizations

## 🚨 Prerequisites Check

### Required Trained Models

1. **For Ensemble Evaluation**:
   - Trained ensemble models (densenet121, resnet101, densenet201, efficientnet_b4)
   - Models should be in `outputs/model_name/best_ensemble_*/best_model.pth`
   - Or ensemble training summary at `outputs/ensemble_training_summary/best_ensemble_training_summary.json`

2. **For Student Evaluation**:
   - Trained KD student model at:
     `outputs/knowledge_distillation/shufflenet_v2/best_kd_distillation_20250530152128/best_model.pth`
   - Or any recent KD model in `outputs/knowledge_distillation/*/best_kd_*/best_model.pth`

### Dataset Structure

Ensure your test datasets are organized as:
```
tomatovillage/
└── test/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    ├── Tomato___Late_blight/
    └── ... (other classes)

plantvillage/
└── test/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    ├── Tomato___Late_blight/
    └── ... (other classes)
```

## 📝 Usage Examples

### Evaluate Both Models on All Test Sets

```bash
# Evaluate ensemble model on test sets
python evaluate_ensemble_on_test_datasets.py

# Evaluate student model on test sets
python evaluate_kd_on_test_datasets.py
```

### Check Available Test Datasets

```bash
# Check what test datasets are available
ls */test/
```

### View Results

```bash
# View ensemble results
cat outputs/test_evaluation/ensemble/ensemble_on_test_datasets_*/test_evaluation_summary.json

# View student results
cat outputs/test_evaluation/student/student_on_test_datasets_*/test_evaluation_summary.json
```

## 🎯 Research Applications

### Model Validation
- **Original Test Set Performance**: Validate on the intended test data
- **Consistent Performance**: Ensure models perform well across different datasets
- **Production Readiness**: Validate models before deployment

### Efficiency Analysis
- **Size vs Performance**: Quantify exact trade-offs on real test data
- **Deployment Decision**: Choose between ensemble and student based on requirements
- **Knowledge Transfer Effectiveness**: Measure how well KD preserved performance

### Comparative Analysis
- **Ensemble vs Student**: Direct comparison on same test data
- **Dataset Differences**: Compare performance across tomatovillage vs plantvillage
- **Class-wise Analysis**: Identify which diseases are consistently challenging
- **Error Pattern Analysis**: Understanding model limitations and failure modes

## 🔄 Troubleshooting

### Common Issues

1. **No test datasets found**: Ensure test directories exist with proper structure
2. **No trained models found**: Run ensemble training and/or KD training first
3. **Class mapping errors**: Verify test dataset class names match training data
4. **Memory issues**: Reduce batch size in dataloader creation

### Verification Commands

```bash
# Check if models exist
ls outputs/*/best_ensemble_*/best_model.pth
ls outputs/knowledge_distillation/*/best_kd_*/best_model.pth

# Check test datasets
ls */test/
ls tomatovillage/test/
ls plantvillage/test/

# Check results
ls outputs/test_evaluation/
```

### Expected Test Dataset Structure

Run this to verify your test datasets are properly structured:
```bash
# Check tomatovillage test structure
find tomatovillage/test -type d -maxdepth 1 | head -10

# Check plantvillage test structure  
find plantvillage/test -type d -maxdepth 1 | head -10
```

## 📊 Expected Research Insights

After running both evaluations, you'll have:

1. **Original Test Performance**: How your models perform on their intended test data
2. **Cross-Dataset Consistency**: Whether performance is consistent across datasets
3. **Ensemble vs Student Comparison**: Direct performance and efficiency comparison
4. **Class-wise Analysis**: Which diseases are easier/harder across datasets
5. **Production Readiness Metrics**: Real performance metrics for deployment decisions
6. **Knowledge Distillation Effectiveness**: How much performance was retained in compression

### Expected Results Summary

| Metric | Ensemble | Student | Compression Ratio |
|---|---|---|---|
| **F1 Score (tomatovillage)** | ~96-98% | ~92-95% | 140× smaller |
| **F1 Score (plantvillage)** | ~95-97% | ~90-93% | 140× smaller |
| **Accuracy (tomatovillage)** | ~97-99% | ~93-96% | 140× smaller |
| **Accuracy (plantvillage)** | ~96-98% | ~91-94% | 140× smaller |
| **Inference Speed** | ~4× slower | Baseline | 140× smaller |

This evaluation provides definitive validation of your models on the original test data, confirming their real-world applicability and the effectiveness of your knowledge distillation approach. 