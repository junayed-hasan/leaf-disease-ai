"""
Dataset Loading and Balancing for Data Balancing Experiments
Extends the original dataset module with balancing techniques
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
from pathlib import Path
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from rich.console import Console

from dataset import LeafDataset, load_data, get_dataset_config, load_data_single_directory, load_data_split_directory
from config import get_image_size, VAL_SIZE, TEST_SIZE
from data_balancing_config import (
    DataBalancingConfig, 
    OfflineAugmentationDataset
)

console = Console()

class BalancedDataset(Dataset):
    """Dataset wrapper that applies resampling techniques"""
    
    def __init__(self, original_dataset: Dataset, resampled_indices: List[int]):
        self.original_dataset = original_dataset
        self.resampled_indices = resampled_indices
        
    def __len__(self):
        return len(self.resampled_indices)
    
    def __getitem__(self, idx):
        original_idx = self.resampled_indices[idx]
        return self.original_dataset[original_idx]

def get_class_counts(dataset: Dataset) -> Dict[int, int]:
    """Get class distribution from dataset"""
    class_counts = Counter()
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[int(label)] += 1
    return {int(k): int(v) for k, v in class_counts.items()}

def extract_features_for_resampling(dataset: Dataset, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for resampling techniques like SMOTE and ADASYN"""
    console.print("Extracting features for resampling...")
    
    features = []
    labels = []
    
    # Simple feature extraction using resized and flattened images
    for i in range(len(dataset)):
        image, label = dataset[i]
        
        # Convert tensor to numpy and resize if needed
        if isinstance(image, torch.Tensor):
            # If image is already transformed (tensor), convert back
            image_np = image.permute(1, 2, 0).numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = np.array(image)
        
        # Resize to smaller size for SMOTE (computational efficiency)
        import cv2
        resized = cv2.resize(image_np, (32, 32))  # Small size for SMOTE
        
        # Flatten to 1D feature vector
        feature_vector = resized.flatten()
        features.append(feature_vector)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def apply_resampling(dataset: Dataset, technique_name: str, image_size: int) -> Dataset:
    """Apply resampling technique to dataset"""
    console.print(f"Applying {technique_name} resampling...")
    
    # Get original class distribution
    original_counts = get_class_counts(dataset)
    console.print(f"Original class distribution: {original_counts}")
    
    # Extract features for resampling
    X, y = extract_features_for_resampling(dataset, image_size)
    
    # Apply resampling technique
    resampler = DataBalancingConfig.get_resampler(technique_name)
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    # Get new class distribution
    new_counts = Counter(y_resampled)
    console.print(f"Resampled class distribution: {dict(new_counts)}")
    
    # Map resampled indices back to original dataset indices
    resampled_indices = []
    original_indices_by_class = {}
    
    # Group original indices by class
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in original_indices_by_class:
            original_indices_by_class[label] = []
        original_indices_by_class[label].append(i)
    
    # For each resampled sample, find corresponding original index
    class_sample_counters = {label: 0 for label in original_counts.keys()}
    
    for label in y_resampled:
        original_class_indices = original_indices_by_class[label]
        sample_idx_in_class = class_sample_counters[label] % len(original_class_indices)
        original_idx = original_class_indices[sample_idx_in_class]
        resampled_indices.append(original_idx)
        class_sample_counters[label] += 1
    
    return BalancedDataset(dataset, resampled_indices)

def create_offline_augmented_dataset(dataset: Dataset, class_counts: Dict[int, int], 
                                   augmentation_config: Dict, image_size: int) -> Dataset:
    """Create dataset with offline augmentation for balancing"""
    console.print("Creating offline augmented dataset...")
    
    # Get target samples per class (use max class count or specified target)
    max_class_count = max(class_counts.values())
    target_samples = DataBalancingConfig.BALANCING_TECHNIQUES["offline_augmentation"]["params"]["target_samples_per_class"]
    target_samples_per_class = max(max_class_count, target_samples)
    
    console.print(f"Target samples per class: {target_samples_per_class}")
    
    # Always create a complete transform chain for offline augmentation
    # Since augmented images will be PIL images, we need the full pipeline
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    augmented_dataset = OfflineAugmentationDataset(
        original_dataset=dataset,
        class_counts=class_counts,
        target_samples_per_class=target_samples_per_class,
        augmentation_config=augmentation_config,
        transform=transform
    )
    
    # Get new class distribution
    new_counts = get_class_counts(augmented_dataset)
    console.print(f"Augmented class distribution: {new_counts}")
    
    return augmented_dataset

def load_balanced_data(dataset_name: str, model_name: str, 
                      balancing_technique: Optional[str] = None,
                      augmentation_config: Optional[Dict] = None,
                      batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Load data with specified balancing technique"""
    
    # First load the data using the existing load_data function
    train_loader, val_loader, test_loader, class_to_idx = load_data(
        dataset_name, model_name, augmentation_config
    )
    
    # If no balancing technique is specified, return the original loaders
    if not balancing_technique or balancing_technique == "focal_loss":
        return train_loader, val_loader, test_loader, class_to_idx
    
    # Extract the dataset from the train_loader for balancing
    train_dataset = train_loader.dataset
    
    # Get original class distribution
    original_counts = get_class_counts(train_dataset)
    console.print(f"Original training class distribution: {original_counts}")
    
    # Apply balancing technique to training set only
    if balancing_technique in ["random_oversampling", "smote", "adasyn"]:
        # Get image size for model
        image_size = get_image_size(model_name)
        train_dataset = apply_resampling(train_dataset, balancing_technique, image_size)
        
    elif balancing_technique == "offline_augmentation":
        # For offline augmentation, we need to work with raw images before transforms
        # Create a raw dataset without transforms
        dataset_config = get_dataset_config(dataset_name)
        if dataset_config['type'] == 'single_directory':
            X_train, _, _, y_train, _, _, _ = load_data_single_directory(
                dataset_config['root'], VAL_SIZE, TEST_SIZE
            )
        else:
            X_train, _, _, y_train, _, _, _ = load_data_split_directory(
                dataset_config['train_dir'], dataset_config['val_dir'], dataset_config['test_dir']
            )
        
        # Create raw dataset without transforms
        raw_train_dataset = LeafDataset(X_train, y_train, transform=None)
        
        # Create offline augmented dataset
        if augmentation_config is None:
            console.print("[yellow]Warning: No augmentation config provided for offline augmentation[/yellow]")
            augmentation_config = {
                "horizontal_flip": {"p": 0.5},
                "rotation": {"degrees": 20},
                "brightness": {"brightness": 0.1}
            }
        
        # Get image size for model
        image_size = get_image_size(model_name)
        train_dataset = create_offline_augmented_dataset(
            raw_train_dataset, original_counts, augmentation_config, image_size
        )
    
    # Create new train data loader with balanced dataset
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, class_to_idx

def analyze_balanced_dataset(dataset_name: str, balancing_technique: Optional[str], 
                           save_dir: Path) -> Dict[str, Any]:
    """Analyze dataset after balancing"""
    
    # Load a sample dataset to get statistics using existing load_data function
    train_loader, _, _, _ = load_data(dataset_name, "default", None)
    sample_dataset = train_loader.dataset
    
    original_counts = get_class_counts(sample_dataset)
    
    stats = {
        "dataset": dataset_name,
        "balancing_technique": balancing_technique,
        "original_class_distribution": original_counts,
        "total_original_samples": sum(original_counts.values()),
        "num_classes": len(original_counts),
        "imbalance_ratio": max(original_counts.values()) / min(original_counts.values())
    }
    
    # If balancing technique is applied, estimate new distribution
    if balancing_technique:
        if balancing_technique in ["random_oversampling", "smote", "adasyn"]:
            # These techniques balance to majority class
            majority_count = max(original_counts.values())
            balanced_counts = {cls: majority_count for cls in original_counts.keys()}
            stats["balanced_class_distribution"] = balanced_counts
            stats["total_balanced_samples"] = sum(balanced_counts.values())
            
        elif balancing_technique == "offline_augmentation":
            target_per_class = DataBalancingConfig.BALANCING_TECHNIQUES["offline_augmentation"]["params"]["target_samples_per_class"]
            max_original = max(original_counts.values())
            target_samples = max(max_original, target_per_class)
            balanced_counts = {cls: target_samples for cls in original_counts.keys()}
            stats["balanced_class_distribution"] = balanced_counts
            stats["total_balanced_samples"] = sum(balanced_counts.values())
    
    return stats 