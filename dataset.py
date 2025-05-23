"""
Dataset handling and data loading utilities
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

console = Console()

class LeafDataset(Dataset):
    """Custom Dataset for loading leaf disease images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(model_name: str = "default", train: bool = True) -> transforms.Compose:
    """Get transforms for training/validation"""
    image_size = get_image_size(model_name)
    
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

def load_data_single_directory(data_root: Path, val_size: float, test_size: float) -> Tuple[List, List, List, List, List, List, Dict]:
    """Load data from a single directory structure and split into train/val/test"""
    image_paths = []
    labels = []
    class_to_idx = {}
    
    # First, create class mapping
    for idx, class_dir in enumerate(sorted(data_root.iterdir())):
        if class_dir.is_dir():
            class_to_idx[class_dir.name] = idx
            
    # Then collect all images with proper labels
    for class_dir in sorted(data_root.iterdir()):
        if class_dir.is_dir():
            class_idx = class_to_idx[class_dir.name]
            for img_path in class_dir.glob("*.[jJ][pP][gG]"):  # Case-insensitive jpg
                image_paths.append(str(img_path))
                labels.append(class_idx)
            for img_path in class_dir.glob("*.[pP][nN][gG]"):  # Case-insensitive png
                image_paths.append(str(img_path))
                labels.append(class_idx)
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), 
        stratify=y_train_val, random_state=SEED
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx

def load_data_split_directory(train_dir: Path, val_dir: Path, test_dir: Path) -> Tuple[List, List, List, List, List, List, Dict]:
    """Load data from pre-split train/val/test directories"""
    class_to_idx = {}
    # Create class mapping from training directory
    for idx, class_dir in enumerate(sorted(train_dir.iterdir())):
        if class_dir.is_dir():
            class_to_idx[class_dir.name] = idx
    
    def load_split(split_dir: Path) -> Tuple[List, List]:
        images = []
        labels = []
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                class_idx = class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.[jJ][pP][gG]"):
                    images.append(str(img_path))
                    labels.append(class_idx)
                for img_path in class_dir.glob("*.[pP][nN][gG]"):
                    images.append(str(img_path))
                    labels.append(class_idx)
        return images, labels
    
    X_train, y_train = load_split(train_dir)
    X_val, y_val = load_split(val_dir)
    X_test, y_test = load_split(test_dir)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx

def load_data(dataset_name: str, model_name: str = "default") -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Load and prepare data based on dataset configuration"""
    dataset_config = get_dataset_config(dataset_name)
    
    if dataset_config['type'] == 'single_directory':
        X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx = load_data_single_directory(
            dataset_config['root'], VAL_SIZE, TEST_SIZE
        )
    else:  # split_directory
        X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx = load_data_split_directory(
            dataset_config['train_dir'], dataset_config['val_dir'], dataset_config['test_dir']
        )
    
    # Create datasets
    train_dataset = LeafDataset(X_train, y_train, transform=get_transforms(model_name, train=True))
    val_dataset = LeafDataset(X_val, y_val, transform=get_transforms(model_name, train=False))
    test_dataset = LeafDataset(X_test, y_test, transform=get_transforms(model_name, train=False))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_to_idx

def analyze_dataset(dataset_name: str, save_dir: Path) -> Tuple[dict, list]:
    """Analyze and visualize dataset statistics"""
    dataset_config = get_dataset_config(dataset_name)
    class_counts = {}
    image_sizes = []
    
    console.print(f"\n[bold]Analyzing {dataset_config['name']} dataset...[/bold]")
    
    if dataset_config['type'] == 'single_directory':
        # Single directory analysis
        for class_dir in sorted(dataset_config['root'].iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]"))
                class_counts[class_name] = len(images)
                
                # Sample image sizes
                for img_path in images[:10]:
                    with Image.open(img_path) as img:
                        image_sizes.append(img.size)
                        
        total_images = sum(class_counts.values())
        
        # Create table
        table = Table(title=f"{dataset_config['name']} Dataset Statistics")
        table.add_column("Class", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        for class_name, count in class_counts.items():
            percentage = (count / total_images) * 100
            table.add_row(class_name, str(count), f"{percentage:.2f}%")
            
    else:  # split_directory
        splits = {
            'Train': dataset_config['train_dir'],
            'Validation': dataset_config['val_dir'],
            'Test': dataset_config['test_dir']
        }
        class_counts = {split: {} for split in splits}
        
        for split_name, split_dir in splits.items():
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    class_name = class_dir.name
                    images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]"))
                    class_counts[split_name][class_name] = len(images)
                    
                    # Sample image sizes from training set only
                    if split_name == 'Train':
                        for img_path in images[:10]:
                            with Image.open(img_path) as img:
                                image_sizes.append(img.size)
        
        # Create table
        table = Table(title=f"{dataset_config['name']} Dataset Statistics")
        table.add_column("Split", style="cyan")
        table.add_column("Class", style="magenta")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        for split_name, split_counts in class_counts.items():
            total_images = sum(split_counts.values())
            for class_name, count in split_counts.items():
                percentage = (count / total_images) * 100
                table.add_row(split_name, class_name, str(count), f"{percentage:.2f}%")
    
    console.print(table)
    
    # Plot class distribution
    if dataset_config['type'] == 'single_directory':
        plt.figure(figsize=(15, 8))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{dataset_config["name"]} Class Distribution')
        plt.tight_layout()
    else:
        plt.figure(figsize=(20, 8))
        num_classes = len(next(iter(class_counts.values())))
        x = np.arange(num_classes)
        width = 0.25
        
        # Plot bars for each split
        for i, (split_name, split_counts) in enumerate(class_counts.items()):
            plt.bar(x + i * width, 
                   list(split_counts.values()), 
                   width, 
                   label=split_name)
        
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title(f'{dataset_config["name"]} Class Distribution by Split')
        plt.xticks(x + width, list(next(iter(class_counts.values())).keys()), 
                  rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
    
    plt.savefig(save_dir / 'class_distribution.png')
    plt.close()
    
    # Image size statistics
    sizes_df = pd.DataFrame(image_sizes, columns=['width', 'height'])
    console.print("\n[bold]Image Size Statistics:[/bold]")
    console.print(sizes_df.describe().to_string())
    
    # Save detailed statistics
    stats = {
        'class_distribution': class_counts,
        'image_size_stats': sizes_df.describe().to_dict()
    }
    
    return stats, image_sizes

def visualize_samples(dataset_name: str, class_to_idx: Dict, save_dir: Path) -> None:
    """Visualize sample images from each class"""
    dataset_config = get_dataset_config(dataset_name)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Determine source directory for samples
    if dataset_config['type'] == 'single_directory':
        source_dir = dataset_config['root']
    else:
        source_dir = dataset_config['train_dir']
    
    # Calculate grid size based on number of classes
    num_classes = len(class_to_idx)
    grid_size = int(np.ceil(np.sqrt(num_classes)))
    
    plt.figure(figsize=(grid_size * 4, grid_size * 4))
    for idx, class_name in idx_to_class.items():
        class_dir = source_dir / class_name
        sample_images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]"))
        if sample_images:
            sample_image = sample_images[0]  # Take first image
            
            plt.subplot(grid_size, grid_size, idx + 1)
            img = Image.open(sample_image)
            plt.imshow(img)
            plt.title(class_name.replace("___", "\n"), fontsize=8)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Test the data loading pipeline for all datasets
    for dataset_name in ['plantvillage', 'tomatovillage', 'combined']:
        console.print(f"\n[bold]Testing {dataset_name} dataset:[/bold]")
        test_save_dir = BASE_OUTPUT_DIR / f"test_run_{dataset_name}"
        test_save_dir.mkdir(exist_ok=True)
        
        # Analyze dataset
        stats, image_sizes = analyze_dataset(dataset_name, test_save_dir)
        
        # Test data loading
        train_loader, val_loader, test_loader, class_to_idx = load_data(dataset_name)
        
        # Visualize samples
        visualize_samples(dataset_name, class_to_idx, test_save_dir)
        
        # Print basic statistics
        console.print(f"\nNumber of batches in train loader: {len(train_loader)}")
        console.print(f"Number of batches in val loader: {len(val_loader)}")
        console.print(f"Number of batches in test loader: {len(test_loader)}")
        console.print(f"Number of classes: {len(class_to_idx)}")