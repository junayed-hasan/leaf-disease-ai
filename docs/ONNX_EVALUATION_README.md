# ONNX Quantized Model Evaluation System

Complete evaluation system for ONNX quantized models on tomato leaf disease datasets.

## Overview

This system provides comprehensive evaluation of ONNX quantized models using:
- **Combined test dataset** (all datasets combined)
- **Individual test datasets** (TomatoVillage and PlantVillage separately)
- **Performance metrics** (accuracy, F1-score, inference time, model size)
- **Visualization tools** (confusion matrices, comparison charts)

## Prerequisites

1. **ONNX Quantized Model**: Ensure you have created the ONNX quantized model:
   ```bash
   # Run mobile quantization pipeline first
   python mobile_quantization_pipeline.py
   ```
   This should create: `mobile_models/shufflenet_v2_mobile_quantized.onnx`

2. **Dependencies**: Install required packages:
   ```bash
   pip install onnxruntime scikit-learn matplotlib seaborn rich wandb
   ```

## Files Created

### Core Components
- `onnx_model_evaluator.py` - Base ONNX model evaluator class
- `evaluate_onnx_quantized_on_combined_test.py` - Combined dataset evaluation
- `evaluate_onnx_quantized_on_test_datasets.py` - Individual datasets evaluation
- `run_onnx_quantized_evaluation.py` - Comprehensive evaluation runner

## Usage

### Option 1: Quick Comprehensive Evaluation (Recommended)
```bash
python run_onnx_quantized_evaluation.py
```
This runs all evaluations automatically and creates a comprehensive summary.

### Option 2: Individual Evaluations

#### Evaluate on Combined Test Dataset
```bash
python evaluate_onnx_quantized_on_combined_test.py
```

#### Evaluate on Individual Test Datasets
```bash
python evaluate_onnx_quantized_on_test_datasets.py
```

## Output Structure

```
outputs/
├── onnx_quantized_evaluation/           # Comprehensive results
│   └── onnx_evaluation_YYYYMMDDHHMMSS/
│       ├── comprehensive_onnx_evaluation_results.json
│       └── execution_results.json
├── test_evaluation/
│   ├── onnx_quantized_combined/         # Combined test results
│   │   └── onnx_quantized_combined_test_YYYYMMDDHHMMSS/
│   │       ├── evaluation_summary.json
│   │       ├── combined_onnx_quantized_performance_metrics.json
│   │       ├── combined_onnx_quantized_confusion_matrix.png
│   │       └── combined_onnx_quantized_classification_report.json
│   └── onnx_quantized_individual/       # Individual test results
│       └── onnx_quantized_individual_test_YYYYMMDDHHMMSS/
│           ├── evaluation_summary.json
│           ├── tomatovillage_onnx_quantized_performance_metrics.json
│           ├── plantvillage_onnx_quantized_performance_metrics.json
│           ├── onnx_quantized_comparative_analysis.png
│           └── onnx_quantized_comparative_summary.json
```

## Key Features

### 1. ONNX Runtime Optimization
- CPU-optimized inference with ONNX Runtime
- Graph optimization enabled
- Quantized model support

### 2. Comprehensive Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1-scores
- **Precision/Recall**: Per-class and averaged
- **Inference Time**: Average time per batch
- **Model Size**: Quantized model size in MB

### 3. Visualization
- **Confusion Matrices**: Visual classification results
- **Comparative Charts**: Performance across datasets
- **Per-class Analysis**: Detailed class-wise metrics

### 4. Error Handling
- Robust error handling for model loading
- Fallback mechanisms for edge cases
- Detailed error reporting

## Expected Performance

### Model Characteristics
- **Format**: ONNX Quantized
- **Runtime**: CPU (onnxruntime)
- **Size**: ~2-5 MB (vs ~11 MB original)
- **Accuracy Retention**: 90-95% of original model

### Typical Results
```
Combined Test Dataset:
  Accuracy: 92-96%
  F1-Score (Macro): 90-94%
  Inference Time: 0.01-0.05s per batch
  Model Size: 2-5 MB

Individual Datasets:
  TomatoVillage: 90-95% accuracy
  PlantVillage: 93-97% accuracy
```

## Integration with Existing Scripts

The ONNX evaluation system is designed to work alongside your existing evaluation scripts:

- Uses the same `test_dataset_loader.py` for individual datasets
- Compatible with existing class mappings
- Follows same output structure patterns
- Integrates with WandB for experiment tracking

## Troubleshooting

### Common Issues

1. **"No ONNX quantized model found"**
   ```bash
   # Solution: Run mobile quantization first
   python mobile_quantization_pipeline.py
   ```

2. **"onnxruntime not installed"**
   ```bash
   pip install onnxruntime
   ```

3. **Class mapping issues**
   - The system handles class mapping automatically
   - Uses same splits as training (VAL_SIZE=0.15, TEST_SIZE=0.15, SEED=42)

4. **Memory issues**
   - ONNX quantized models run on CPU with lower memory usage
   - Batch processing handles large datasets efficiently

### Performance Optimization

1. **CPU Optimization**: ONNX Runtime automatically optimizes for CPU
2. **Batch Size**: Default batch size works well for most systems
3. **Graph Optimization**: Enabled by default for better performance

## Technical Details

### ONNX Model Loading
```python
# Example usage of the evaluator
from onnx_model_evaluator import ONNXModelEvaluator

evaluator = ONNXModelEvaluator("mobile_models/shufflenet_v2_mobile_quantized.onnx")
predictions, targets, probabilities, inference_time = evaluator.predict(test_loader)
```

### Class Mapping Compatibility
- Automatically maps between combined (15 classes) and individual datasets
- TomatoVillage: 8 classes
- PlantVillage: 10 classes
- Combined: 15 classes

### Inference Pipeline
1. Load ONNX model with optimized session
2. Convert PyTorch tensors to numpy arrays
3. Run ONNX inference on CPU
4. Convert results back to PyTorch format
5. Calculate metrics and generate reports

## Next Steps

1. **Run Evaluation**: Start with comprehensive evaluation
2. **Analyze Results**: Check accuracy retention vs original model
3. **Deploy Model**: Use ONNX model for mobile deployment
4. **Compare Performance**: Compare with other quantization methods

## Support

For issues or questions:
1. Check the execution logs in output directories
2. Review error messages in `execution_results.json`
3. Verify ONNX model exists and is valid
4. Ensure all dependencies are installed correctly 