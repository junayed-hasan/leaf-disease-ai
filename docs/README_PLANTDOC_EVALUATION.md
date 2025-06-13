# PlantDoc Dataset Evaluation

This module provides scripts to evaluate your trained ensemble and knowledge distillation models on the external PlantDoc dataset. The PlantDoc dataset contains a subset of tomato disease classes that map to your original combined dataset classes.

## 🎯 Overview

The PlantDoc evaluation system:
1. **Maps PlantDoc classes** to your original combined dataset classes
2. **Evaluates both models**: Ensemble and Knowledge Distillation student model
3. **Provides comprehensive analysis**: Performance metrics, confusion matrices, misclassification analysis
4. **Follows the same pipeline structure** as your training scripts

## 📊 Class Mapping

The PlantDoc dataset contains 8 tomato classes that map to your combined dataset:

| PlantDoc Class | Combined Dataset Class | Description |
|---|---|---|
| `Tomato Early blight leaf` | `Tomato___Early_blight` | Early blight disease |
| `Tomato leaf` | `Tomato___healthy` | Healthy tomato leaves |
| `Tomato leaf bacterial spot` | `Tomato___Bacterial_spot` | Bacterial spot disease |
| `Tomato leaf late blight` | `Tomato___Late_blight` | Late blight disease |
| `Tomato leaf mosaic virus` | `Tomato___Tomato_mosaic_virus` | Mosaic virus disease |
| `Tomato leaf yellow virus` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Yellow curl virus |
| `Tomato mold leaf` | `Tomato___Leaf_Mold` | Leaf mold disease |
| `Tomato Septoria leaf spot` | `Tomato___Septoria_leaf_spot` | Septoria leaf spot |

## 📁 Files Overview

### Core Scripts

1. **`plantdoc_dataset.py`** - Dataset loader and class mapping utilities
2. **`evaluate_ensemble_on_plantdoc.py`** - Evaluates your trained ensemble model
3. **`evaluate_kd_on_plantdoc.py`** - Evaluates your trained student model from KD

### Key Features

- ✅ **Automatic Class Mapping**: Maps PlantDoc classes to original model classes
- ✅ **WandB Integration**: Full experiment tracking and logging
- ✅ **Dataset Analysis**: Complete statistics and sample visualization  
- ✅ **Model Loading**: Robust loading of trained ensemble and KD models
- ✅ **Comprehensive Evaluation**: Detailed metrics, confusion matrices, misclassification analysis
- ✅ **Results Saving**: All outputs saved in organized directory structure

## 🚀 Quick Start

### Prerequisites

1. **Trained Models**: Either ensemble models or KD student model (or both)
2. **PlantDoc Dataset**: In `plantdoc/` directory with the class structure shown above
3. **Dependencies**: All packages from your existing requirements files

### Step 1: Evaluate Ensemble Model

```bash
python evaluate_ensemble_on_plantdoc.py
```

**What it does:**
- Finds your trained ensemble models automatically
- Loads the PlantDoc dataset with proper class mapping
- Evaluates ensemble using soft voting
- Generates comprehensive performance analysis
- Saves confusion matrix, classification report, and misclassification analysis

**Expected Time**: ~30-60 minutes

### Step 2: Evaluate Knowledge Distillation Student Model

```bash
python evaluate_kd_on_plantdoc.py
```

**What it does:**
- Finds your trained KD student model (from the path you provided)
- Loads the PlantDoc dataset with proper class mapping
- Evaluates student model performance
- Compares efficiency (model size vs performance)
- Saves detailed performance analysis

**Expected Time**: ~15-30 minutes

## 📁 Output Structure

```
outputs/
└── plantdoc_evaluation/
    ├── ensemble/
    │   └── ensemble_on_plantdoc_YYYYMMDDHHMMSS/
    │       ├── config.json
    │       ├── plantdoc_dataset_statistics.json
    │       ├── plantdoc_sample_visualizations.png
    │       ├── plantdoc_performance_metrics.json
    │       ├── plantdoc_classification_report.json
    │       ├── plantdoc_confusion_matrix.png
    │       ├── plantdoc_misclassifications.json
    │       └── evaluation_summary.json
    └── student/
        └── student_on_plantdoc_YYYYMMDDHHMMSS/
            ├── config.json
            ├── plantdoc_dataset_statistics.json
            ├── plantdoc_sample_visualizations.png
            ├── plantdoc_student_performance_metrics.json
            ├── plantdoc_student_classification_report.json
            ├── plantdoc_student_confusion_matrix.png
            ├── plantdoc_student_misclassifications.json
            └── evaluation_summary.json
```

## 🔧 Pipeline Flow Details

### Ensemble Evaluation (`evaluate_ensemble_on_plantdoc.py`)

1. **Initialize WandB** - Project: `tomato-disease-plantdoc-evaluation`
2. **Load Class Mapping** - Original combined dataset class mapping
3. **Find Ensemble Models** - Automatically locate trained ensemble models
4. **Analyze PlantDoc Dataset** - Statistics and class distribution
5. **Visualize Samples** - Sample images with class mapping
6. **Load PlantDoc Data** - Create DataLoader with class mapping
7. **Create Ensemble** - Load all models and create soft voting ensemble
8. **Get Predictions** - Ensemble predictions on PlantDoc data
9. **Evaluate Performance** - Comprehensive metrics and analysis
10. **Save Results** - Detailed performance analysis and visualizations

### Student Model Evaluation (`evaluate_kd_on_plantdoc.py`)

1. **Initialize WandB** - Project: `tomato-disease-plantdoc-evaluation`
2. **Find KD Model** - Locate trained student model from KD
3. **Load Class Mapping** - Original combined dataset class mapping
4. **Analyze PlantDoc Dataset** - Statistics and class distribution
5. **Visualize Samples** - Sample images with class mapping
6. **Load PlantDoc Data** - Create DataLoader with model-specific image size
7. **Create Student Evaluator** - Load student model
8. **Get Predictions** - Student model predictions on PlantDoc data
9. **Evaluate Performance** - Metrics including model efficiency analysis
10. **Save Results** - Detailed performance analysis and comparisons

## 📈 Expected Performance

### Performance Comparison

| Model Type | Parameters | Expected F1 Score | Speed |
|---|---|---|---|
| **Ensemble** | ~200M | 85-95% | Slow (4 models) |
| **Student (KD)** | ~1.4M | 80-90% | Fast (1 model) |

### Key Metrics Tracked

- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro)**: Unweighted average F1 across classes
- **F1 Score (Weighted)**: Sample-weighted average F1
- **Per-Class Metrics**: Precision, Recall, F1 for each class
- **Misclassification Analysis**: Detailed error analysis
- **Model Efficiency**: Parameters vs performance trade-off

## 🔍 Key Features

### Automatic Model Discovery
- **Ensemble Models**: Automatically finds trained ensemble models from training summary
- **KD Student Model**: Finds your specific KD model or most recent one
- **Fallback Search**: Robust search mechanisms if summaries not found

### Robust Class Mapping
- **Automatic Mapping**: PlantDoc classes mapped to original model classes
- **Subset Handling**: Properly handles subset of classes present in PlantDoc
- **Validation**: Verifies all mappings before evaluation

### Comprehensive Analysis
- **Performance Metrics**: All standard classification metrics
- **Visual Analysis**: Confusion matrices and sample visualizations
- **Error Analysis**: Detailed misclassification analysis with confidence scores
- **Efficiency Analysis**: Model size vs performance trade-offs

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
   - Trained KD student model at your specified path:
     `outputs/knowledge_distillation/shufflenet_v2/best_kd_distillation_20250530152128/best_model.pth`
   - Or any recent KD model in `outputs/knowledge_distillation/*/best_kd_*/best_model.pth`

### Dataset Structure

Ensure your PlantDoc dataset is organized as:
```
plantdoc/
├── Tomato Early blight leaf/
├── Tomato leaf/
├── Tomato leaf bacterial spot/
├── Tomato leaf late blight/
├── Tomato leaf mosaic virus/
├── Tomato leaf yellow virus/
├── Tomato mold leaf/
└── Tomato Septoria leaf spot/
```

## 📝 Usage Examples

### Evaluate Both Models

```bash
# Evaluate ensemble model
python evaluate_ensemble_on_plantdoc.py

# Evaluate student model  
python evaluate_kd_on_plantdoc.py
```

### Check Specific Results

```bash
# View ensemble results
cat outputs/plantdoc_evaluation/ensemble/ensemble_on_plantdoc_*/evaluation_summary.json

# View student results
cat outputs/plantdoc_evaluation/student/student_on_plantdoc_*/evaluation_summary.json
```

## 🎯 Research Applications

### Model Validation
- **External Dataset Validation**: Test model generalization on unseen data
- **Cross-Dataset Performance**: Compare performance across different datasets
- **Domain Adaptation**: Assess model robustness to different image sources

### Efficiency Analysis
- **Size vs Performance**: Quantify the trade-off between model size and accuracy
- **Deployment Readiness**: Evaluate models for production deployment
- **Knowledge Transfer**: Assess effectiveness of knowledge distillation

### Comparative Analysis
- **Ensemble vs Student**: Direct comparison of large ensemble vs compact student
- **Class-wise Performance**: Identify which diseases are harder to classify
- **Error Analysis**: Understanding failure modes and potential improvements

## 🔄 Troubleshooting

### Common Issues

1. **No trained models found**: Run ensemble training and/or KD training first
2. **PlantDoc dataset not found**: Ensure dataset is in correct directory structure
3. **Class mapping errors**: Verify PlantDoc class names match expected format
4. **Memory issues**: Reduce batch size in dataloader creation

### Verification Commands

```bash
# Check if models exist
ls outputs/*/best_ensemble_*/best_model.pth
ls outputs/knowledge_distillation/*/best_kd_*/best_model.pth

# Check PlantDoc dataset
ls plantdoc/

# Check results
ls outputs/plantdoc_evaluation/
```

## 📊 Expected Research Insights

After running both evaluations, you'll have:

1. **Performance Comparison**: Direct comparison between ensemble and student models
2. **Efficiency Analysis**: Trade-off between model size and performance  
3. **External Validation**: How well your models generalize to new data
4. **Class-wise Analysis**: Which diseases are easier/harder to classify
5. **Deployment Readiness**: Performance metrics for production deployment decisions

This evaluation provides crucial validation of your trained models on external data, demonstrating their real-world applicability and efficiency for tomato disease classification tasks. 