"""
Data Balancing Configuration for Systematic Experiments
Implements various techniques to handle class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import cv2
from pathlib import Path

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DataBalancingConfig:
    """Configuration class for data balancing experiments"""
    
    # Best hyperparameters from previous experiments
    BEST_HYPERPARAMETERS = {
        "learning_rate": 0.0005,
        "scheduler": "step",
        "optimizer": "adam", 
        "weight_decay": 0.0001
    }
    
    # Data balancing techniques
    BALANCING_TECHNIQUES = {
        "random_oversampling": {
            "name": "Random Oversampling",
            "type": "resampling",
            "params": {
                "random_state": 42,
                "sampling_strategy": "auto"  # Balance all classes to majority
            }
        },
        "smote": {
            "name": "SMOTE",
            "type": "resampling", 
            "params": {
                "random_state": 42,
                "k_neighbors": 5,
                "sampling_strategy": "auto"
            }
        },
        "adasyn": {
            "name": "ADASYN",
            "type": "resampling",
            "params": {
                "random_state": 42,
                "n_neighbors": 5,
                "sampling_strategy": "auto"
            }
        },
        "offline_augmentation": {
            "name": "Offline Augmentation",
            "type": "augmentation",
            "params": {
                "target_samples_per_class": 2000,  # Target number of samples per class
                "augmentation_factor": 3  # How many augmented versions per original image
            }
        },
        "focal_loss": {
            "name": "Focal Loss",
            "type": "loss_function",
            "params": {
                "alpha": None,  # Will be computed based on class frequencies
                "gamma": 2.0
            }
        }
    }
    
    @classmethod
    def get_focal_loss(cls, class_counts: Dict[int, int], device: str = 'cuda') -> FocalLoss:
        """Create focal loss with class-based alpha weighting"""
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        # Compute alpha weights (inverse frequency)
        alpha_weights = []
        for class_idx in range(num_classes):
            class_freq = class_counts.get(class_idx, 1)
            alpha = total_samples / (num_classes * class_freq)
            alpha_weights.append(alpha)
        
        alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32).to(device)
        return FocalLoss(alpha=alpha_tensor, gamma=2.0)
    
    @classmethod
    def get_resampler(cls, technique_name: str):
        """Get resampling technique instance"""
        if technique_name == "random_oversampling":
            params = cls.BALANCING_TECHNIQUES[technique_name]["params"]
            return RandomOverSampler(**params)
        elif technique_name == "smote":
            params = cls.BALANCING_TECHNIQUES[technique_name]["params"]
            return SMOTE(**params)
        elif technique_name == "adasyn":
            params = cls.BALANCING_TECHNIQUES[technique_name]["params"]
            return ADASYN(**params)
        else:
            raise ValueError(f"Unknown resampling technique: {technique_name}")

class OfflineAugmentationDataset(Dataset):
    """Dataset that applies offline augmentation for balancing"""
    
    def __init__(self, original_dataset: Dataset, class_counts: Dict[int, int], 
                 target_samples_per_class: int, augmentation_config: Dict,
                 transform=None):
        self.original_dataset = original_dataset
        self.class_counts = class_counts
        self.target_samples_per_class = target_samples_per_class
        self.augmentation_config = augmentation_config
        self.transform = transform
        
        # Generate augmented indices
        self.augmented_indices = self._generate_augmented_indices()
        
    def _generate_augmented_indices(self) -> List[Tuple[int, int]]:
        """Generate list of (original_idx, augmentation_version) pairs"""
        indices = []
        
        # Group samples by class
        class_to_indices = {}
        for idx in range(len(self.original_dataset)):
            _, label = self.original_dataset[idx]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # For each class, generate augmented samples
        for class_label, original_indices in class_to_indices.items():
            current_count = len(original_indices)
            needed_samples = max(0, self.target_samples_per_class - current_count)
            
            # Add original samples
            for idx in original_indices:
                indices.append((idx, 0))  # 0 means original, no augmentation
            
            # Add augmented samples
            if needed_samples > 0:
                for _ in range(needed_samples):
                    # Randomly select an original sample to augment
                    original_idx = np.random.choice(original_indices)
                    aug_version = np.random.randint(1, 6)  # Random augmentation version
                    indices.append((original_idx, aug_version))
        
        return indices
    
    def __len__(self):
        return len(self.augmented_indices)
    
    def __getitem__(self, idx):
        original_idx, aug_version = self.augmented_indices[idx]
        image, label = self.original_dataset[original_idx]
        
        # Apply augmentation if needed
        if aug_version > 0:
            image = self._apply_augmentation(image, aug_version)
        
        # Apply final transform
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def _apply_augmentation(self, image, aug_version):
        """Apply specific augmentation based on version"""
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
        
        # Apply augmentations based on config and version
        if aug_version == 1 and "horizontal_flip" in self.augmentation_config:
            if np.random.random() < self.augmentation_config["horizontal_flip"]["p"]:
                image = cv2.flip(image, 1)
        
        if aug_version == 2 and "rotation" in self.augmentation_config:
            angle = np.random.uniform(-self.augmentation_config["rotation"]["degrees"], 
                                    self.augmentation_config["rotation"]["degrees"])
            center = (image.shape[1]//2, image.shape[0]//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        
        if aug_version == 3 and "brightness" in self.augmentation_config:
            brightness_factor = 1.0 + np.random.uniform(-self.augmentation_config["brightness"]["brightness"],
                                                       self.augmentation_config["brightness"]["brightness"])
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Add more augmentation versions as needed
        if aug_version >= 4:
            # Combine multiple augmentations
            if np.random.random() < 0.5 and "horizontal_flip" in self.augmentation_config:
                image = cv2.flip(image, 1)
            if np.random.random() < 0.5 and "rotation" in self.augmentation_config:
                angle = np.random.uniform(-10, 10)
                center = (image.shape[1]//2, image.shape[0]//2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        
        # Convert back to tensor format
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        
        return image

class DataBalancingExperiments:
    """Define systematic data balancing experiments"""
    
    @classmethod
    def get_all_experiments(cls) -> Dict[str, Dict]:
        """Get all data balancing experiments"""
        experiments = {}
        
        # 1. Baseline (no balancing)
        experiments["baseline"] = {
            "name": "Baseline (No Balancing)",
            "group": "baseline",
            "technique": None,
            "loss_function": "cross_entropy",
            "description": "Baseline experiment with class imbalance"
        }
        
        # 2. Resampling techniques
        for technique_name, technique_config in DataBalancingConfig.BALANCING_TECHNIQUES.items():
            if technique_config["type"] == "resampling":
                exp_name = f"resample_{technique_name}"
                experiments[exp_name] = {
                    "name": technique_config["name"],
                    "group": "resampling",
                    "technique": technique_name,
                    "loss_function": "cross_entropy",
                    "description": f"Resampling with {technique_config['name']}"
                }
        
        # 3. Offline augmentation
        experiments["offline_aug"] = {
            "name": "Offline Augmentation",
            "group": "augmentation",
            "technique": "offline_augmentation",
            "loss_function": "cross_entropy", 
            "description": "Pre-generated augmented samples for balancing"
        }
        
        # 4. Focal loss (no resampling)
        experiments["focal_loss"] = {
            "name": "Focal Loss",
            "group": "loss_function",
            "technique": "focal_loss",
            "loss_function": "focal_loss",
            "description": "Focal loss for handling class imbalance"
        }
        
        # 5. Combined approaches - All resampling techniques with focal loss
        experiments["random_oversampling_focal"] = {
            "name": "Random Oversampling + Focal Loss",
            "group": "combined",
            "technique": "random_oversampling",
            "loss_function": "focal_loss",
            "description": "Random oversampling with focal loss"
        }
        
        experiments["smote_focal"] = {
            "name": "SMOTE + Focal Loss",
            "group": "combined",
            "technique": "smote",
            "loss_function": "focal_loss",
            "description": "SMOTE resampling with focal loss"
        }
        
        experiments["adasyn_focal"] = {
            "name": "ADASYN + Focal Loss",
            "group": "combined",
            "technique": "adasyn",
            "loss_function": "focal_loss",
            "description": "ADASYN resampling with focal loss"
        }
        
        experiments["offline_aug_focal"] = {
            "name": "Offline Augmentation + Focal Loss",
            "group": "combined", 
            "technique": "offline_augmentation",
            "loss_function": "focal_loss",
            "description": "Offline augmentation with focal loss"
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
        return ["baseline", "resampling", "augmentation", "loss_function", "combined"] 