# Installation Guide

This guide provides step-by-step instructions for setting up TomatoLeaf-AI on your system.

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Intel i5/AMD Ryzen 5 or better (for training: Intel i7/AMD Ryzen 7+)
- **RAM**: 8GB minimum (16GB+ recommended for training)
- **Storage**: 50GB free space (datasets require ~20GB)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (if using GPU)
- **Git**: For cloning the repository

## 🚀 Quick Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/junayed-hasan/lightweight-tomato.git
cd lightweight-tomato

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Option 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/junayed-hasan/lightweight-tomato.git
cd lightweight-tomato

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_balancing.txt

# Install development dependencies
pip install -e ".[dev]"
```

## 🔧 Detailed Installation Steps

### Step 1: Environment Setup

#### Using Conda (Recommended)
```bash
# Create conda environment
conda create -n tomatoleaf python=3.9
conda activate tomatoleaf

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
pip install -r requirements_balancing.txt
```

#### Using pip + virtualenv
```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create virtual environment
virtualenv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
pip install -r requirements_balancing.txt
```

### Step 2: Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test package installation
python -c "from src.models import *; print('Models imported successfully')"
python -c "from src.datasets import *; print('Datasets imported successfully')"
```

### Step 3: Download Datasets

#### Automatic Download (Coming Soon)
```bash
# Download and setup datasets automatically
python scripts/download_datasets.py
```

#### Manual Download
1. **PlantVillage Dataset**:
   - Download from [Kaggle PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
   - Extract to `data/plantvillage/`

2. **TomatoVillage Dataset**:
   - Download from [TomatoVillage Repository](https://github.com/your-repo/tomatovillage)
   - Extract to `data/tomatovillage/`

3. **Create Combined Dataset**:
   ```bash
   python src/datasets/dataset.py --create_combined
   ```

## 🐳 Docker Installation

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/junayed-hasan/lightweight-tomato.git
cd lightweight-tomato

# Build and run with Docker Compose
docker-compose up --build
```

### Manual Docker Setup

```bash
# Build Docker image
docker build -t tomatoleaf-ai .

# Run container
docker run -it --gpus all -v $(pwd):/workspace tomatoleaf-ai
```

### Dockerfile
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements_balancing.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_balancing.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

CMD ["bash"]
```

## 📱 Mobile App Setup

### Flutter Installation

1. **Install Flutter SDK**:
   ```bash
   # Download Flutter SDK
   git clone https://github.com/flutter/flutter.git -b stable
   export PATH="$PATH:`pwd`/flutter/bin"
   
   # Verify installation
   flutter doctor
   ```

2. **Setup Mobile App**:
   ```bash
   cd mobile_app/
   flutter pub get
   flutter run
   ```

### Android Setup
```bash
# Install Android Studio and SDK
# Set environment variables
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/tools
export PATH=$PATH:$ANDROID_HOME/tools/bin
export PATH=$PATH:$ANDROID_HOME/platform-tools

# Accept licenses
flutter doctor --android-licenses
```

### iOS Setup (macOS only)
```bash
# Install Xcode from App Store
# Install CocoaPods
sudo gem install cocoapods

# Setup iOS dependencies
cd mobile_app/ios/
pod install
```

## 🔍 Troubleshooting

### Common Issues

#### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
```bash
# Reduce batch size in configuration
# Edit src/configurations/config.py
BATCH_SIZE = 16  # Reduce from 32

# Use gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 2
```

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/lightweight-tomato"
```

### Platform-Specific Issues

#### Windows
```bash
# Use Windows-compatible paths
# Install Microsoft Visual C++ Build Tools
# Use PowerShell or Command Prompt as Administrator
```

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew dependencies
brew install python@3.9
```

#### Linux
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential

# For GPU support
sudo apt-get install nvidia-driver-470 nvidia-cuda-toolkit
```

## ✅ Verification Tests

### Basic Functionality Test
```bash
# Run basic tests
python -m pytest tests/test_basic.py -v

# Test model loading
python -c "
from src.models.model_factory import create_model
model = create_model('densenet121', num_classes=15)
print('Model created successfully')
"

# Test dataset loading
python -c "
from src.datasets.dataset import TomatoDataset
dataset = TomatoDataset('data/combined', split='train')
print(f'Dataset loaded: {len(dataset)} samples')
"
```

### Training Test
```bash
# Quick training test (1 epoch)
python scripts/train.py \
    --model densenet121 \
    --epochs 1 \
    --batch_size 8 \
    --data_dir data/combined
```

### Evaluation Test
```bash
# Test evaluation pipeline
python src/evaluation/evaluate_kd_on_test_datasets.py \
    --model_path checkpoints/sample_model.pth \
    --test_only
```

## 📚 Additional Resources

### Documentation
- [Project Overview](PROJECT_OVERVIEW.md)
- [API Documentation](API_REFERENCE.md)
- [Training Guide](TRAINING_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

### Community
- [GitHub Issues](https://github.com/junayed-hasan/lightweight-tomato/issues)
- [Discussions](https://github.com/junayed-hasan/lightweight-tomato/discussions)
- [Contributing Guide](../CONTRIBUTING.md)

### Support
If you encounter issues not covered in this guide:
1. Check the [troubleshooting section](#troubleshooting)
2. Search [existing issues](https://github.com/junayed-hasan/lightweight-tomato/issues)
3. Create a [new issue](https://github.com/junayed-hasan/lightweight-tomato/issues/new) with detailed information

---

**Next Steps**: After successful installation, proceed to the [Training Guide](TRAINING_GUIDE.md) to start training models. 