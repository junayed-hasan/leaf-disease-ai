# TomatoLeaf-AI Docker Image
# Multi-stage build for optimized production image

# Stage 1: Base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel as base

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development image
FROM base as development

# Copy requirements first for better caching
COPY requirements.txt requirements_balancing.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_balancing.txt

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=6.0 \
    pytest-cov>=2.0 \
    black>=21.0 \
    flake8>=3.8 \
    isort>=5.0 \
    jupyter \
    ipykernel

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p data/combined data/plantvillage data/tomatovillage \
    outputs/models outputs/logs outputs/visualizations \
    checkpoints/ensemble checkpoints/distilled checkpoints/quantized

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command
CMD ["bash"]

# Stage 3: Production image (lightweight)
FROM base as production

# Copy only requirements
COPY requirements.txt ./

# Install only production dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY setup.py ./

# Install package
RUN pip install .

# Create non-root user for security
RUN useradd -m -u 1000 tomatoleaf && \
    chown -R tomatoleaf:tomatoleaf /workspace

USER tomatoleaf

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; import src.models; print('OK')" || exit 1

# Default command for production
CMD ["python", "scripts/train.py", "--help"]

# Stage 4: Mobile/Edge deployment image
FROM python:3.9-slim as mobile

WORKDIR /workspace

# Install minimal dependencies for mobile deployment
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    onnx \
    onnxruntime \
    numpy \
    pillow \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy only quantization and inference code
COPY src/quantization/ ./src/quantization/
COPY src/models/ ./src/models/
COPY src/utils/ ./src/utils/

# Copy mobile-specific scripts
COPY scripts/inference.py ./

# Set environment variables for CPU-only inference
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1

# Default command for mobile inference
CMD ["python", "inference.py"] 