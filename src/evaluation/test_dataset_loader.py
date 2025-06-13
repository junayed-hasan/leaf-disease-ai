#!/usr/bin/env python3
"""
Test Dataset Loader for Original Datasets
Loads test sets from the same programmatic splits used during original model training
Maps individual dataset classes to combined dataset classes for evaluation
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from rich.console import Console

# Import existing modules
from config import *
from dataset import load_data_single_directory, load_data_split_directory, get_dataset_config

console = Console()

# Class mappings from individual datasets to combined dataset
TOMATOVILLAGE_TO_COMBINED_MAPPING = {
    "Early_blight": "Tomato___Early_blight",
    "Healthy": "Tomato___healthy", 
    "Late_blight": "Tomato___Late_blight",
    "Leaf Miner": "Tomato___Leaf_Miner",
    "Magnesium Deficiency": "Tomato___Magnesium_Deficiency",
    "Nitrogen Deficiency": "Tomato___Nitrogen_Deficiency",
    "Pottassium Deficiency": "Tomato___Pottassium_Deficiency",
    "Spotted Wilt Virus": "Tomato___Spotted_Wilt_Virus"
}

PLANTVILLAGE_TO_COMBINED_MAPPING = {
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight", 
    "Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot": "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy": "Tomato___healthy"
}

class TestDatasetFromSplits(Dataset):
    """Test dataset using the same splits as original training"""
    
    def __init__(self, dataset_name: str, combined_class_mapping: Dict[str, int], 
                 transform: Optional[transforms.Compose] = None):
        self.dataset_name = dataset_name
        self.combined_class_mapping = combined_class_mapping
        self.transform = transform
        
        # Get dataset configuration
        self.dataset_config = get_dataset_config(dataset_name)
        
        # Get class mapping for this individual dataset
        if dataset_name.lower() == "tomatovillage":
            self.class_mapping = TOMATOVILLAGE_TO_COMBINED_MAPPING
        elif dataset_name.lower() == "plantvillage":
            self.class_mapping = PLANTVILLAGE_TO_COMBINED_MAPPING
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Load test split using the same method as original training
        self.test_samples = []
        self.class_counts = {}
        
        self._load_test_split()
        
        console.print(f"[bold]{dataset_name} Test Split loaded:[/bold]")
        console.print(f"  Total test samples: {len(self.test_samples)}")
        console.print(f"  Individual dataset classes: {len(self.class_counts)}")
        console.print(f"  Test class distribution:")
        for class_name, count in self.class_counts.items():
            mapped_class = self.class_mapping.get(class_name, "UNMAPPED")
            console.print(f"    {class_name} -> {mapped_class}: {count} samples")
    
    def _load_test_split(self):
        """Load test split using the same method as original dataset.py"""
        
        # Load data using the same split method as original training
        if self.dataset_config['type'] == 'single_directory':
            X_train, X_val, X_test, y_train, y_val, y_test, original_class_to_idx = load_data_single_directory(
                self.dataset_config['root'], VAL_SIZE, TEST_SIZE
            )
        else:  # split_directory
            X_train, X_val, X_test, y_train, y_val, y_test, original_class_to_idx = load_data_split_directory(
                self.dataset_config['train_dir'], self.dataset_config['val_dir'], self.dataset_config['test_dir']
            )
        
        # Create reverse mapping from original dataset indices to class names
        original_idx_to_class = {v: k for k, v in original_class_to_idx.items()}
        
        # Process test split
        for img_path, original_label in zip(X_test, y_test):
            original_class_name = original_idx_to_class[original_label]
            
            # Map to combined class
            if original_class_name not in self.class_mapping:
                console.print(f"[yellow]Warning: Class '{original_class_name}' not found in mapping, skipping[/yellow]")
                continue
                
            combined_class_name = self.class_mapping[original_class_name]
            
            # Check if combined class exists in target mapping
            if combined_class_name not in self.combined_class_mapping:
                console.print(f"[yellow]Warning: Combined class '{combined_class_name}' not found in target mapping, skipping[/yellow]")
                continue
            
            combined_label = self.combined_class_mapping[combined_class_name]
            
            # Count samples
            if original_class_name not in self.class_counts:
                self.class_counts[original_class_name] = 0
            self.class_counts[original_class_name] += 1
            
            # Add to test samples
            self.test_samples.append({
                'image_path': str(img_path),
                'original_class_name': original_class_name,
                'combined_class_name': combined_class_name,
                'combined_label': combined_label,
                'dataset': self.dataset_name
            })
    
    def __len__(self):
        return len(self.test_samples)
    
    def __getitem__(self, idx):
        sample = self.test_samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['combined_label'], sample
    
    def get_dataset_info(self):
        """Get detailed dataset information"""
        
        # Convert Path objects to strings for JSON serialization
        dataset_config_serializable = {}
        for key, value in self.dataset_config.items():
            if isinstance(value, Path):
                dataset_config_serializable[key] = str(value)
            else:
                dataset_config_serializable[key] = value
        
        dataset_info = {
            'dataset_name': self.dataset_name,
            'total_test_samples': len(self.test_samples),
            'num_original_classes': len(self.class_counts),
            'original_class_counts': self.class_counts,
            'class_mapping': self.class_mapping,
            'dataset_config': dataset_config_serializable
        }
        return dataset_info

def create_test_dataloader(dataset_name: str, combined_class_mapping: Dict[str, int], 
                          image_size: int = 224, batch_size: int = 32) -> DataLoader:
    """Create DataLoader for test dataset using original splits"""
    
    def custom_collate_fn(batch):
        """Custom collate function to handle sample dictionaries"""
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        samples = [item[2] for item in batch]  # Keep as list
        return images, labels, samples
    
    # Define transforms (same as original model training)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = TestDatasetFromSplits(
        dataset_name=dataset_name,
        combined_class_mapping=combined_class_mapping,
        transform=transform
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=0,  # Reduce to 0 to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return dataloader, dataset

def analyze_test_dataset(dataset_name: str, combined_class_mapping: Dict[str, int], save_dir: Path):
    """Analyze the test dataset and save statistics"""
    
    console.print(f"[bold]Analyzing {dataset_name} Test Dataset[/bold]")
    
    # Create test dataset
    dataset = TestDatasetFromSplits(dataset_name, combined_class_mapping)
    dataset_info = dataset.get_dataset_info()
    
    # Save statistics
    with open(save_dir / f"{dataset_name}_test_dataset_statistics.json", "w") as f:
        json.dump(dataset_info, f, indent=4)
    
    return dataset_info

def visualize_test_samples(dataset_name: str, combined_class_mapping: Dict[str, int], 
                          save_dir: Path, samples_per_class: int = 3):
    """Visualize sample images from test dataset"""
    
    import matplotlib.pyplot as plt
    
    console.print(f"[bold]Creating {dataset_name} test sample visualizations[/bold]")
    
    # Create test dataset
    dataset = TestDatasetFromSplits(dataset_name, combined_class_mapping)
    
    if len(dataset.test_samples) == 0:
        console.print(f"[yellow]No test samples found for {dataset_name}[/yellow]")
        return
    
    # Get class samples
    class_samples = {}
    for sample in dataset.test_samples:
        original_class = sample['original_class_name']
        combined_class = sample['combined_class_name']
        display_name = f"{original_class}\n→ {combined_class}"
        
        if display_name not in class_samples:
            class_samples[display_name] = []
        if len(class_samples[display_name]) < samples_per_class:
            class_samples[display_name].append(sample['image_path'])
    
    if not class_samples:
        console.print(f"[yellow]No class samples to visualize for {dataset_name}[/yellow]")
        return
    
    # Create visualization
    num_classes = len(class_samples)
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                           figsize=(samples_per_class * 3, num_classes * 3))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    elif samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    class_idx = 0
    for class_name, sample_paths in class_samples.items():
        for sample_idx, img_path in enumerate(sample_paths):
            if sample_idx >= samples_per_class:
                break
                
            try:
                img = Image.open(img_path).convert('RGB')
                if num_classes == 1 and samples_per_class == 1:
                    axes.imshow(img)
                    axes.axis('off')
                    axes.set_title(class_name, fontsize=8)
                elif num_classes == 1:
                    axes[sample_idx].imshow(img)
                    axes[sample_idx].axis('off')
                    if sample_idx == 0:
                        axes[sample_idx].set_ylabel(class_name, rotation=0, ha='right', va='center', fontsize=8)
                elif samples_per_class == 1:
                    axes[class_idx].imshow(img)
                    axes[class_idx].axis('off')
                    axes[class_idx].set_ylabel(class_name, rotation=0, ha='right', va='center', fontsize=8)
                else:
                    axes[class_idx, sample_idx].imshow(img)
                    axes[class_idx, sample_idx].axis('off')
                    
                    if sample_idx == 0:
                        axes[class_idx, sample_idx].set_ylabel(
                            class_name, rotation=0, ha='right', va='center', fontsize=6
                        )
                        
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load {img_path}: {e}[/yellow]")
                if num_classes > 1 and samples_per_class > 1:
                    axes[class_idx, sample_idx].axis('off')
        
        # Hide unused subplots in this row
        if num_classes > 1 and samples_per_class > 1:
            for sample_idx in range(len(sample_paths), samples_per_class):
                axes[class_idx, sample_idx].axis('off')
            
        class_idx += 1
    
    plt.suptitle(f'{dataset_name} Test Dataset Samples with Class Mapping', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_dir / f"{dataset_name}_test_sample_visualizations.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Test sample visualizations saved to: {save_dir / f'{dataset_name}_test_sample_visualizations.png'}[/green]")

def get_available_test_datasets():
    """Get list of available test datasets based on dataset configurations"""
    available_datasets = []
    
    # Check which datasets are available based on directory existence
    for dataset_name in ['tomatovillage', 'plantvillage']:
        try:
            dataset_config = get_dataset_config(dataset_name)
            
            if dataset_config['type'] == 'single_directory':
                if dataset_config['root'].exists():
                    available_datasets.append(dataset_name)
            else:  # split_directory
                if (dataset_config['train_dir'].exists() and 
                    dataset_config['val_dir'].exists() and 
                    dataset_config['test_dir'].exists()):
                    available_datasets.append(dataset_name)
                    
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check {dataset_name}: {e}[/yellow]")
            continue
    
    return available_datasets

def get_dataset_class_subset(dataset_name: str, combined_class_mapping: Dict[str, int]) -> Dict[str, int]:
    """Get the subset of combined classes that exist in the individual dataset"""
    
    if dataset_name.lower() == "tomatovillage":
        class_mapping = TOMATOVILLAGE_TO_COMBINED_MAPPING
    elif dataset_name.lower() == "plantvillage":
        class_mapping = PLANTVILLAGE_TO_COMBINED_MAPPING
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataset_subset = {}
    
    for original_class, combined_class in class_mapping.items():
        if combined_class in combined_class_mapping:
            dataset_subset[combined_class] = combined_class_mapping[combined_class]
    
    return dataset_subset 