"""
Configuration file containing all hyperparameters and settings
"""
from pathlib import Path
import os
from datetime import datetime
from typing import Optional, Dict

# Dataset configurations
DATASET_CONFIGS = {
    'plantvillage': {
        'name': 'PlantVillage',
        'root': Path("plantVillage/color"),
        'num_classes': 10,
        'type': 'single_directory'  # All data in one directory, needs splitting
    },
    'tomatovillage': {
        'name': 'TomatoVillage',
        'root': Path("tomatovillage/Variant-a(Multiclass Classification)"),
        'train_dir': Path("tomatovillage/Variant-a(Multiclass Classification)/train"),
        'val_dir': Path("tomatovillage/Variant-a(Multiclass Classification)/val"),
        'test_dir': Path("tomatovillage/Variant-a(Multiclass Classification)/test"),
        'num_classes': 8,
        'type': 'split_directory'  # Pre-split into train/val/test
    },
    'combined': {
        'name': 'Combined',
        'root': Path("combined"),
        'num_classes': 15,  # Total number of unique classes after combining
        'type': 'single_directory'  # All data in one directory, needs splitting
    }
}

# Output directory setup
BASE_OUTPUT_DIR = Path("outputs")
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

def get_dataset_config(dataset_name: str) -> Dict:
    """Get configuration for specific dataset"""
    return DATASET_CONFIGS[dataset_name.lower()]

def get_run_dir(model_name: str, dataset_name: str, experiment_name: Optional[str] = None) -> Path:
    """Get the output directory for the current run"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    exp_name = experiment_name if experiment_name else f"baseline_{dataset_name}_{model_name}"
    run_dir = BASE_OUTPUT_DIR / model_name / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

# Model-specific settings
MODEL_CONFIGS = {
    'inception_v3': {'image_size': 299},
    'xception': {'image_size': 299},
    'efficientnet_b4': {'image_size': 380},
    'swin_large': {'image_size': 192},
    'default': {'image_size': 224}
}

def get_image_size(model_name: str) -> int:
    """Get the appropriate image size for a given model"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['default'])['image_size']

# Training settings
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Model settings
PRETRAINED = True

# Logging settings
LOG_INTERVAL = 10

# Scheduler settings
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1
MIN_LR = 1e-6

# Device settings
DEVICE = "cuda"  # or "cpu"