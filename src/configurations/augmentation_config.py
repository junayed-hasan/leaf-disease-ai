"""
Augmentation Configuration for Systematic Experiments
Designed for scientific research with clear ablation studies
"""

import torch
from torchvision import transforms
from typing import Dict, List, Tuple, Optional
import torchvision.transforms.functional as TF
import random
import numpy as np

class GaussianBlur:
    """Custom Gaussian Blur transform"""
    def __init__(self, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return TF.gaussian_blur(img, self.kernel_size, sigma)

class GaussianNoise:
    """Add Gaussian noise to tensor images (applied after ToTensor)"""
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor_img):
        """Apply Gaussian noise to tensor image"""
        if not isinstance(tensor_img, torch.Tensor):
            raise TypeError(f"GaussianNoise expects tensor input, got {type(tensor_img)}")
        
        noise = torch.randn_like(tensor_img) * self.std + self.mean
        return torch.clamp(tensor_img + noise, 0.0, 1.0)

class AugmentationConfig:
    """Configuration class for systematic augmentation experiments"""
    
    # Base transforms (always applied)
    BASE_TRANSFORMS = [
        "Resize",
        "ToTensor", 
        "Normalize"
    ]
    
    # Augmentation groups for ablation studies
    GEOMETRIC_TRANSFORMS = {
        "horizontal_flip": {
            "name": "RandomHorizontalFlip",
            "ranges": {"p": [0.5]}  # Standard probability
        },
        "vertical_flip": {
            "name": "RandomVerticalFlip", 
            "ranges": {"p": [0.5]}  # Standard probability
        },
        "rotation": {
            "name": "RandomRotation",
            "ranges": {"degrees": [10, 20, 30]}  # ±10°, ±20°, ±30°
        },
        "scale_crop": {
            "name": "RandomResizedCrop",
            "ranges": {"scale": [(0.9, 1.1), (0.8, 1.2), (0.7, 1.3)]}  # Different zoom ranges
        }
    }
    
    PHOTOMETRIC_TRANSFORMS = {
        "brightness": {
            "name": "ColorJitter",
            "ranges": {"brightness": [0.1, 0.2, 0.3]}  # ±10%, ±20%, ±30%
        },
        "contrast": {
            "name": "ColorJitter", 
            "ranges": {"contrast": [0.1, 0.2, 0.3]}
        },
        "saturation": {
            "name": "ColorJitter",
            "ranges": {"saturation": [0.1, 0.2, 0.3]}
        },
        "hue": {
            "name": "ColorJitter",
            "ranges": {"hue": [0.05, 0.1, 0.15]}  # Smaller range for hue
        },
        "color_combined": {
            "name": "ColorJitter",
            "ranges": {
                "brightness": [0.2],
                "contrast": [0.2], 
                "saturation": [0.2],
                "hue": [0.1]
            }
        }
    }
    
    NOISE_BLUR_TRANSFORMS = {
        "gaussian_blur": {
            "name": "GaussianBlur",
            "ranges": {"sigma": [(0.1, 1.0), (0.1, 2.0), (0.1, 3.0)]}  # Light to moderate blur
        },
        "gaussian_noise": {
            "name": "GaussianNoise", 
            "ranges": {"std": [0.02, 0.05, 0.1]}  # Light to moderate noise
        }
    }
    
    @classmethod
    def get_baseline_transforms(cls, image_size: int, train: bool = True) -> transforms.Compose:
        """Get baseline transforms without any augmentation"""
        if train:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    @classmethod
    def get_augmented_transforms(cls, image_size: int, augmentation_config: Dict, train: bool = True) -> transforms.Compose:
        """Get transforms with specified augmentations"""
        if not train:
            return cls.get_baseline_transforms(image_size, train=False)
        
        transform_list = []
        tensor_transforms = []  # Transforms that need tensor input
        
        # Always start with resize
        transform_list.append(transforms.Resize((image_size, image_size)))
        
        # Separate PIL-based and tensor-based augmentations
        for aug_name, aug_params in augmentation_config.items():
            transform = cls._create_transform(aug_name, aug_params)
            if transform:
                # Gaussian noise needs to be applied after ToTensor
                if aug_name == "gaussian_noise":
                    tensor_transforms.append(transform)
                else:
                    transform_list.append(transform)
        
        # Add ToTensor conversion
        transform_list.append(transforms.ToTensor())
        
        # Add tensor-based augmentations (like gaussian noise)
        transform_list.extend(tensor_transforms)
        
        # Always end with Normalize
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225]))
        
        return transforms.Compose(transform_list)
    
    @classmethod
    def _create_transform(cls, aug_name: str, aug_params: Dict):
        """Create a specific transform based on name and parameters"""
        if aug_name == "horizontal_flip":
            return transforms.RandomHorizontalFlip(p=aug_params.get("p", 0.5))
        
        elif aug_name == "vertical_flip":
            return transforms.RandomVerticalFlip(p=aug_params.get("p", 0.5))
        
        elif aug_name == "rotation":
            degrees = aug_params.get("degrees", 10)
            return transforms.RandomRotation(degrees)
        
        elif aug_name == "scale_crop":
            scale = aug_params.get("scale", (0.8, 1.2))
            # We'll use RandomResizedCrop with the original size as target
            return transforms.RandomResizedCrop(224, scale=scale)  # Will be resized again later
        
        elif aug_name in ["brightness", "contrast", "saturation", "hue", "color_combined"]:
            return transforms.ColorJitter(
                brightness=aug_params.get("brightness", 0),
                contrast=aug_params.get("contrast", 0),
                saturation=aug_params.get("saturation", 0),
                hue=aug_params.get("hue", 0)
            )
        
        elif aug_name == "gaussian_blur":
            sigma = aug_params.get("sigma", (0.1, 2.0))
            return GaussianBlur(sigma=sigma)
        
        elif aug_name == "gaussian_noise":
            std = aug_params.get("std", 0.05)
            return GaussianNoise(std=std)
        
        return None

class AugmentationExperiments:
    """Define systematic experiments for research paper"""
    
    @classmethod
    def get_all_experiments(cls) -> Dict[str, Dict]:
        """Get all augmentation experiments organized for research paper"""
        experiments = {}
        
        # 1. Baseline (no augmentation)
        experiments["baseline"] = {
            "name": "Baseline (No Augmentation)",
            "group": "baseline",
            "augmentations": {},
            "description": "No data augmentation applied"
        }
        
        # 2. Individual augmentation analysis
        # 2a. Geometric transforms (individual)
        for transform_name, config in AugmentationConfig.GEOMETRIC_TRANSFORMS.items():
            for param_name, values in config["ranges"].items():
                for value in values:
                    exp_name = f"geo_{transform_name}_{param_name}_{value}"
                    experiments[exp_name] = {
                        "name": f"Geometric - {transform_name.replace('_', ' ').title()} ({param_name}={value})",
                        "group": "geometric_individual",
                        "augmentations": {transform_name: {param_name: value}},
                        "description": f"Single geometric augmentation: {transform_name} with {param_name}={value}"
                    }
        
        # 2b. Photometric transforms (individual)
        for transform_name, config in AugmentationConfig.PHOTOMETRIC_TRANSFORMS.items():
            if transform_name == "color_combined":
                continue  # Skip combined for individual analysis
            for param_name, values in config["ranges"].items():
                for value in values:
                    exp_name = f"photo_{transform_name}_{param_name}_{value}"
                    experiments[exp_name] = {
                        "name": f"Photometric - {transform_name.replace('_', ' ').title()} ({param_name}={value})",
                        "group": "photometric_individual", 
                        "augmentations": {transform_name: {param_name: value}},
                        "description": f"Single photometric augmentation: {transform_name} with {param_name}={value}"
                    }
        
        # 2c. Noise/Blur transforms (individual)
        for transform_name, config in AugmentationConfig.NOISE_BLUR_TRANSFORMS.items():
            for param_name, values in config["ranges"].items():
                for value in values:
                    exp_name = f"noise_{transform_name}_{param_name}_{str(value).replace('(', '').replace(')', '').replace(', ', '_')}"
                    experiments[exp_name] = {
                        "name": f"Noise/Blur - {transform_name.replace('_', ' ').title()} ({param_name}={value})",
                        "group": "noise_individual",
                        "augmentations": {transform_name: {param_name: value}},
                        "description": f"Single noise/blur augmentation: {transform_name} with {param_name}={value}"
                    }
        
        # 3. Group-wise combinations (best parameters from individual analysis)
        # These would typically be determined after individual analysis
        experiments.update({
            "geometric_combined": {
                "name": "Geometric Combined (Best Parameters)",
                "group": "geometric_combined",
                "augmentations": {
                    "horizontal_flip": {"p": 0.5},
                    "rotation": {"degrees": 20},  # Assuming 20° performs best
                    "scale_crop": {"scale": (0.8, 1.2)}  # Assuming this range performs best
                },
                "description": "Combined geometric augmentations with optimal parameters"
            },
            
            "photometric_combined": {
                "name": "Photometric Combined (Best Parameters)", 
                "group": "photometric_combined",
                "augmentations": {
                    "color_combined": {
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "saturation": 0.2,
                        "hue": 0.1
                    }
                },
                "description": "Combined photometric augmentations with optimal parameters"
            },
            
            "noise_combined": {
                "name": "Noise/Blur Combined (Best Parameters)",
                "group": "noise_combined", 
                "augmentations": {
                    "gaussian_blur": {"sigma": (0.1, 2.0)},  # Assuming this performs best
                    "gaussian_noise": {"std": 0.05}  # Assuming this performs best
                },
                "description": "Combined noise/blur augmentations with optimal parameters"
            }
        })
        
        # 4. Two-group combinations
        experiments.update({
            "geo_photo": {
                "name": "Geometric + Photometric",
                "group": "two_group_combination",
                "augmentations": {
                    "horizontal_flip": {"p": 0.5},
                    "rotation": {"degrees": 20},
                    "scale_crop": {"scale": (0.8, 1.2)},
                    "color_combined": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}
                },
                "description": "Combination of geometric and photometric augmentations"
            },
            
            "geo_noise": {
                "name": "Geometric + Noise/Blur",
                "group": "two_group_combination",
                "augmentations": {
                    "horizontal_flip": {"p": 0.5},
                    "rotation": {"degrees": 20},
                    "scale_crop": {"scale": (0.8, 1.2)},
                    "gaussian_blur": {"sigma": (0.1, 2.0)},
                    "gaussian_noise": {"std": 0.05}
                },
                "description": "Combination of geometric and noise/blur augmentations"
            },
            
            "photo_noise": {
                "name": "Photometric + Noise/Blur",
                "group": "two_group_combination",
                "augmentations": {
                    "color_combined": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
                    "gaussian_blur": {"sigma": (0.1, 2.0)},
                    "gaussian_noise": {"std": 0.05}
                },
                "description": "Combination of photometric and noise/blur augmentations"
            }
        })
        
        # 5. All three groups combined
        experiments["all_combined"] = {
            "name": "All Augmentations Combined",
            "group": "all_combined",
            "augmentations": {
                "horizontal_flip": {"p": 0.5},
                "rotation": {"degrees": 20},
                "scale_crop": {"scale": (0.8, 1.2)},
                "color_combined": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
                "gaussian_blur": {"sigma": (0.1, 2.0)},
                "gaussian_noise": {"std": 0.05}
            },
            "description": "All augmentation groups combined with optimal parameters"
        }
        
        return experiments
    
    @classmethod
    def get_experiments_by_group(cls, group_name: str) -> Dict[str, Dict]:
        """Get experiments filtered by group for organized execution"""
        all_experiments = cls.get_all_experiments()
        return {k: v for k, v in all_experiments.items() if v["group"] == group_name}
    
    @classmethod
    def get_experiment_groups(cls) -> List[str]:
        """Get list of all experiment groups in logical order"""
        return [
            "baseline",
            "geometric_individual", 
            "photometric_individual",
            "noise_individual",
            "geometric_combined",
            "photometric_combined", 
            "noise_combined",
            "two_group_combination",
            "all_combined"
        ] 