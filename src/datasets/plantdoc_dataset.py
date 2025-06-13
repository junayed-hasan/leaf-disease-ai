#!/usr/bin/env python3
"""
PlantDoc Dataset Loader and Class Mapping
Maps plantdoc classes to the original combined dataset classes for model evaluation
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
from rich.console import Console

console = Console()

# Mapping from PlantDoc classes to Combined dataset classes
PLANTDOC_TO_COMBINED_MAPPING = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf": "Tomato___healthy", 
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot"
}

class PlantDocDataset(Dataset):
    """PlantDoc dataset with class mapping to combined dataset"""
    
    def __init__(self, root_dir: str, combined_class_mapping: Dict[str, int], 
                 transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.combined_class_mapping = combined_class_mapping
        self.transform = transform
        
        # Create reverse mapping for combined classes
        self.combined_idx_to_class = {v: k for k, v in combined_class_mapping.items()}
        
        # Load images and create mappings
        self.samples = []
        self.plantdoc_classes = []
        self.class_counts = {}
        
        self._load_dataset()
        
        console.print(f"[bold]PlantDoc Dataset loaded:[/bold]")
        console.print(f"  Total samples: {len(self.samples)}")
        console.print(f"  PlantDoc classes: {len(self.plantdoc_classes)}")
        console.print(f"  Class distribution:")
        for class_name, count in self.class_counts.items():
            mapped_class = PLANTDOC_TO_COMBINED_MAPPING.get(class_name, "UNMAPPED")
            console.print(f"    {class_name} -> {mapped_class}: {count} samples")
    
    def _load_dataset(self):
        """Load all images and create class mappings"""
        
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name not in PLANTDOC_TO_COMBINED_MAPPING:
                console.print(f"[yellow]Warning: Class '{class_name}' not found in mapping, skipping[/yellow]")
                continue
            
            if class_name not in self.plantdoc_classes:
                self.plantdoc_classes.append(class_name)
            
            # Count samples in this class
            image_files = [f for f in class_dir.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            self.class_counts[class_name] = len(image_files)
            
            # Add samples
            for img_path in image_files:
                combined_class = PLANTDOC_TO_COMBINED_MAPPING[class_name]
                combined_label = self.combined_class_mapping[combined_class]
                
                self.samples.append({
                    'image_path': str(img_path),
                    'plantdoc_class': class_name,
                    'combined_class': combined_class,
                    'combined_label': combined_label
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['combined_label'], sample
    
    def get_class_mapping_info(self):
        """Get detailed class mapping information"""
        mapping_info = {
            'plantdoc_to_combined': PLANTDOC_TO_COMBINED_MAPPING,
            'plantdoc_classes': self.plantdoc_classes,
            'class_counts': self.class_counts,
            'total_samples': len(self.samples)
        }
        return mapping_info

def create_plantdoc_dataloader(plantdoc_root: str, combined_class_mapping: Dict[str, int], 
                              image_size: int = 224, batch_size: int = 32) -> DataLoader:
    """Create DataLoader for PlantDoc dataset"""
    
    def custom_collate_fn(batch):
        """Custom collate function to handle sample dictionaries"""
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        samples = [item[2] for item in batch]  # Keep as list, don't try to batch dictionaries
        return images, labels, samples
    
    # Define transforms (same as original model training)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = PlantDocDataset(
        root_dir=plantdoc_root,
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
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    return dataloader, dataset

def analyze_plantdoc_dataset(plantdoc_root: str, save_dir: Path):
    """Analyze the PlantDoc dataset and save statistics"""
    
    console.print("[bold]Analyzing PlantDoc Dataset[/bold]")
    
    # Get basic statistics
    stats = {
        'dataset_name': 'PlantDoc',
        'root_directory': str(plantdoc_root),
        'class_mapping': PLANTDOC_TO_COMBINED_MAPPING,
        'classes': {},
        'total_samples': 0
    }
    
    plantdoc_path = Path(plantdoc_root)
    
    for class_dir in plantdoc_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        if class_name not in PLANTDOC_TO_COMBINED_MAPPING:
            continue
            
        # Count images in this class
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        num_samples = len(image_files)
        stats['classes'][class_name] = {
            'num_samples': num_samples,
            'mapped_to': PLANTDOC_TO_COMBINED_MAPPING[class_name]
        }
        stats['total_samples'] += num_samples
        
        console.print(f"  {class_name}: {num_samples} samples -> {PLANTDOC_TO_COMBINED_MAPPING[class_name]}")
    
    console.print(f"Total samples: {stats['total_samples']}")
    
    # Save statistics
    with open(save_dir / "plantdoc_dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    return stats

def get_plantdoc_subset_classes(combined_class_mapping: Dict[str, int]) -> Dict[str, int]:
    """Get the subset of combined classes that exist in PlantDoc"""
    
    plantdoc_subset = {}
    
    for plantdoc_class, combined_class in PLANTDOC_TO_COMBINED_MAPPING.items():
        if combined_class in combined_class_mapping:
            plantdoc_subset[combined_class] = combined_class_mapping[combined_class]
    
    return plantdoc_subset

def visualize_plantdoc_samples(plantdoc_root: str, save_dir: Path, samples_per_class: int = 3):
    """Visualize sample images from each PlantDoc class"""
    
    import matplotlib.pyplot as plt
    
    console.print("[bold]Creating PlantDoc sample visualizations[/bold]")
    
    plantdoc_path = Path(plantdoc_root)
    
    # Determine grid size
    num_classes = len(PLANTDOC_TO_COMBINED_MAPPING)
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                           figsize=(samples_per_class * 3, num_classes * 3))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    class_idx = 0
    for class_dir in sorted(plantdoc_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in PLANTDOC_TO_COMBINED_MAPPING:
            continue
            
        class_name = class_dir.name
        mapped_class = PLANTDOC_TO_COMBINED_MAPPING[class_name]
        
        # Get sample images
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        sample_files = image_files[:samples_per_class]
        
        for sample_idx, img_path in enumerate(sample_files):
            if sample_idx >= samples_per_class:
                break
                
            try:
                img = Image.open(img_path).convert('RGB')
                axes[class_idx, sample_idx].imshow(img)
                axes[class_idx, sample_idx].axis('off')
                
                if sample_idx == 0:
                    axes[class_idx, sample_idx].set_ylabel(
                        f"{class_name}\n→ {mapped_class}", 
                        rotation=0, ha='right', va='center', fontsize=8
                    )
                    
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load {img_path}: {e}[/yellow]")
                axes[class_idx, sample_idx].axis('off')
        
        # Hide unused subplots in this row
        for sample_idx in range(len(sample_files), samples_per_class):
            axes[class_idx, sample_idx].axis('off')
            
        class_idx += 1
    
    plt.suptitle('PlantDoc Dataset Samples with Class Mapping', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_dir / "plantdoc_sample_visualizations.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Sample visualizations saved to: {save_dir / 'plantdoc_sample_visualizations.png'}[/green]") 