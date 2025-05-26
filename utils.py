"""
Utility functions for the tomato leaf disease classification project
"""
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_dir(model_name: str, dataset_name: str, experiment_name: Optional[str] = None) -> Path:
    """Get the output directory for the current run"""
    BASE_OUTPUT_DIR = Path("outputs")
    BASE_OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    exp_name = experiment_name if experiment_name else f"baseline_{dataset_name}_{model_name}"
    run_dir = BASE_OUTPUT_DIR / model_name / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir 