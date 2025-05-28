"""
Hyperparameter Configuration for Systematic Experiments
Designed for scientific research with clear ablation studies
"""

import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Dict, List, Any, Optional

class HyperparameterConfig:
    """Configuration class for systematic hyperparameter experiments"""
    
    # Hyperparameter categories for ablation studies
    LEARNING_RATES = {
        "lr_1e2": 1e-2,
        "lr_1e3": 1e-3, 
        "lr_5e4": 5e-4,
        "lr_1e4": 1e-4,
        "lr_2e3": 2e-3
    }
    
    SCHEDULERS = {
        "cosine": {
            "name": "CosineAnnealingLR",
            "params": {"T_max": 50, "eta_min": 1e-6}
        },
        "step": {
            "name": "StepLR", 
            "params": {"step_size": 15, "gamma": 0.1}
        },
        "plateau": {
            "name": "ReduceLROnPlateau",
            "params": {"mode": "min", "patience": 5, "factor": 0.5, "min_lr": 1e-6}
        }
    }
    
    OPTIMIZERS = {
        "adam": {
            "name": "Adam",
            "params": {}
        },
        "adamw": {
            "name": "AdamW", 
            "params": {}
        },
        "sgd": {
            "name": "SGD",
            "params": {"momentum": 0.9}
        }
    }
    
    WEIGHT_DECAYS = {
        "wd_1e4": 1e-4,
        "wd_1e3": 1e-3,
        "wd_1e2": 1e-2
    }
    
    @classmethod
    def get_optimizer(cls, optimizer_name: str, model_parameters, learning_rate: float, weight_decay: float):
        """Create optimizer instance"""
        if optimizer_name == "adam":
            return Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    @classmethod
    def get_scheduler(cls, scheduler_name: str, optimizer, epochs: int):
        """Create scheduler instance"""
        if scheduler_name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_name == "step":
            return StepLR(optimizer, step_size=15, gamma=0.1)
        elif scheduler_name == "plateau":
            return ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

class HyperparameterExperiments:
    """Define systematic hyperparameter experiments"""
    
    @classmethod
    def get_all_experiments(cls) -> Dict[str, Dict]:
        """Get all hyperparameter experiments organized by category"""
        experiments = {}
        
        # 1. Learning Rate Experiments
        for lr_name, lr_value in HyperparameterConfig.LEARNING_RATES.items():
            exp_name = f"lr_{lr_name}"
            experiments[exp_name] = {
                "name": f"Learning Rate - {lr_value}",
                "group": "learning_rate",
                "hyperparameters": {
                    "learning_rate": lr_value,
                    "optimizer": "adam",  # Default
                    "scheduler": "plateau",  # Default
                    "weight_decay": 1e-4  # Default
                },
                "description": f"Learning rate ablation: {lr_value}"
            }
        
        # 2. Scheduler Experiments
        for sched_name, sched_config in HyperparameterConfig.SCHEDULERS.items():
            exp_name = f"sched_{sched_name}"
            experiments[exp_name] = {
                "name": f"Scheduler - {sched_config['name']}",
                "group": "scheduler",
                "hyperparameters": {
                    "learning_rate": 1e-3,  # Default
                    "optimizer": "adam",  # Default
                    "scheduler": sched_name,
                    "weight_decay": 1e-4  # Default
                },
                "description": f"Learning rate scheduler: {sched_config['name']}"
            }
        
        # 3. Optimizer Experiments
        for opt_name, opt_config in HyperparameterConfig.OPTIMIZERS.items():
            exp_name = f"opt_{opt_name}"
            experiments[exp_name] = {
                "name": f"Optimizer - {opt_config['name']}",
                "group": "optimizer", 
                "hyperparameters": {
                    "learning_rate": 1e-3,  # Default
                    "optimizer": opt_name,
                    "scheduler": "plateau",  # Default
                    "weight_decay": 1e-4  # Default
                },
                "description": f"Optimizer: {opt_config['name']}"
            }
        
        # 4. Weight Decay Experiments
        for wd_name, wd_value in HyperparameterConfig.WEIGHT_DECAYS.items():
            exp_name = f"wd_{wd_name}"
            experiments[exp_name] = {
                "name": f"Weight Decay - {wd_value}",
                "group": "weight_decay",
                "hyperparameters": {
                    "learning_rate": 1e-3,  # Default
                    "optimizer": "adam",  # Default
                    "scheduler": "plateau",  # Default
                    "weight_decay": wd_value
                },
                "description": f"Weight decay: {wd_value}"
            }
        
        return experiments
    
    @classmethod
    def get_experiments_by_group(cls, group_name: str) -> Dict[str, Dict]:
        """Get experiments for a specific group"""
        all_experiments = cls.get_all_experiments()
        return {name: config for name, config in all_experiments.items() 
                if config["group"] == group_name}
    
    @classmethod
    def get_experiment_groups(cls) -> List[str]:
        """Get list of all experiment groups"""
        return ["learning_rate", "scheduler", "optimizer", "weight_decay"] 