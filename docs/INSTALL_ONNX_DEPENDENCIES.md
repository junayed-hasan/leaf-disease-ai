# ONNX Quantized Model Evaluation - Dependency Installation Guide

## For GCP Compute Engine with Python 3.7.12

This guide helps you install the necessary dependencies for running ONNX quantized model evaluation.

### Required Dependencies

```bash
# Core ONNX Runtime (CPU version - works with quantized models)
pip install onnxruntime==1.15.1

# ONNX for model information (optional but recommended)
pip install onnx==1.14.1

# Core ML dependencies (should already be installed)
pip install scikit-learn matplotlib seaborn

# Rich for better console output (optional)
pip install rich

# WandB for experiment tracking (optional)
pip install wandb
```

### Python 3.7.12 Compatible Versions

If you encounter version conflicts, use these specific versions:

```bash
pip install onnxruntime==1.15.1
pip install onnx==1.14.1
pip install scikit-learn==1.0.2
pip install matplotlib==3.5.3
pip install seaborn==0.11.2
pip install rich==12.6.0
pip install wandb==0.13.10
```

### Minimal Installation (Just for ONNX)

If you want to run only the ONNX evaluation:

```bash
# Essential for ONNX model loading and inference
pip install onnxruntime==1.15.1

# For metrics calculation
pip install scikit-learn==1.0.2

# For visualization
pip install matplotlib==3.5.3 seaborn==0.11.2
```

### Verification

Test if ONNX Runtime is working:

```python
import onnxruntime as ort
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
```

Expected output should include `CPUExecutionProvider`.

### Troubleshooting

1. **"No module named 'onnxruntime'"**
   ```bash
   pip install onnxruntime==1.15.1
   ```

2. **"No module named 'onnx'"** (for model info)
   ```bash
   pip install onnx==1.14.1
   ```

3. **Version conflicts**
   - Use the specific versions listed above
   - Consider using a virtual environment

4. **"rich not available"**
   - This is optional, the script will use basic print statements
   - To install: `pip install rich==12.6.0`

5. **"wandb not available"**
   - This is optional, logging will be skipped
   - To install: `pip install wandb==0.13.10`

### Running Without Optional Dependencies

The scripts are designed to work even if some dependencies are missing:

- **Without rich**: Uses basic print statements
- **Without wandb**: Skips experiment tracking
- **Without onnx**: Limited model info but still works

### Testing Installation

After installation, test with:

```bash
python -c "
import onnxruntime as ort
import sklearn
import matplotlib
import numpy as np
print('✓ All core dependencies installed successfully')
print(f'ONNX Runtime: {ort.__version__}')
"
```

If this runs without errors, you're ready to use the ONNX evaluation scripts. 