#!/usr/bin/env python3
"""
Training Script for Augmentation Experiments
Based on original train.py but with augmentation configuration support
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.console import Console

from config import *
from dataset import load_data, analyze_dataset, visualize_samples, get_dataset_config
from models import ModelFactory
from trainer import Trainer
from utils import set_seed, get_run_dir

console = Console()

def save_config(config: dict, save_dir: Path) -> None:
    """Save experiment configuration"""
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def safe_copytree(src: str, dst: Path) -> None:
    """Safely copy directory tree"""
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not copy {src}: {e}[/yellow]")

def main():
    parser = argparse.ArgumentParser(description='Train model with augmentation experiments')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                      choices=ModelFactory.list_available_models(),
                      help='Model architecture')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='combined',
                      choices=['plantvillage', 'tomatovillage', 'combined'],
                      help='Dataset to use for training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimizer')
    
    # Augmentation arguments
    parser.add_argument('--augmentation_config', type=str, default=None,
                      help='Path to JSON file containing augmentation configuration')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for the experiment')
    
    args = parser.parse_args()
    
    # Load augmentation configuration if provided
    augmentation_config = None
    if args.augmentation_config and Path(args.augmentation_config).exists():
        with open(args.augmentation_config, 'r') as f:
            augmentation_config = json.load(f)
        console.print(f"[bold]Loaded augmentation config from:[/bold] {args.augmentation_config}")
    else:
        console.print("[bold]Using baseline configuration (no augmentation)[/bold]")
    
    dataset_config = get_dataset_config(args.dataset)
    
    # Create experiment name
    if args.experiment_name is None:
        if augmentation_config:
            aug_summary = "_".join([k for k in augmentation_config.keys()])
            args.experiment_name = f"{args.model}_{aug_summary}"
        else:
            args.experiment_name = f"{args.model}_baseline"
    
    # Create run directory
    run_dir = get_run_dir(args.model, args.experiment_name)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Set seed
    set_seed(SEED)
    
    # Get model-specific image size
    image_size = get_image_size(args.model)
    
    # Create config dictionary
    config = {
        "model": args.model,
        "dataset": dataset_config['name'],
        "dataset_type": dataset_config['type'],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "image_size": image_size,
        "pretrained": PRETRAINED,
        "experiment_name": args.experiment_name,
        "output_dir": str(run_dir),
        "augmentation_config": augmentation_config
    }
    
    # Save config
    save_config(config, run_dir)
    
    # Create meaningful run name for wandb
    aug_description = "baseline" if not augmentation_config else "_".join(augmentation_config.keys())
    wandb_name = f"{args.dataset}_{args.model}_{aug_description}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb
    run = wandb.init(
        project="tomato-disease-augmentation",
        name=wandb_name,
        config=config
    )
    
    # Analyze dataset
    console.print("[bold]Step 1: Analyzing Dataset[/bold]")
    stats, image_sizes = analyze_dataset(args.dataset, run_dir)
    
    # Save dataset statistics
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    # Load data with augmentation configuration
    console.print("\n[bold]Step 2: Loading Data[/bold]")
    train_loader, val_loader, test_loader, class_to_idx = load_data(
        args.dataset, args.model, augmentation_config=augmentation_config
    )
    
    # Save class mapping
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    
    # Visualize samples
    console.print("\n[bold]Step 3: Visualizing Samples[/bold]")
    visualize_samples(args.dataset, class_to_idx, run_dir)
    
    # Create model
    console.print("\n[bold]Step 4: Creating Model[/bold]")
    model = ModelFactory.get_model(
        args.model, 
        dataset_config['num_classes'], 
        pretrained=PRETRAINED
    ).to(DEVICE)
    
    # Initialize criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, 
                                 factor=SCHEDULER_FACTOR, min_lr=MIN_LR)
    
    # Print experiment information
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print(f"[bold]Dataset:[/bold] {dataset_config['name']} ({dataset_config['num_classes']} classes)")
    console.print(f"[bold]Image Size:[/bold] {image_size}x{image_size}")
    console.print(f"[bold]Training Configuration:[/bold]")
    console.print(f"  Epochs: {args.epochs}")
    console.print(f"  Batch Size: {args.batch_size}")
    console.print(f"  Learning Rate: {args.learning_rate}")
    console.print(f"  Weight Decay: {WEIGHT_DECAY}")
    
    if augmentation_config:
        console.print(f"[bold]Augmentation Configuration:[/bold]")
        for aug_name, aug_params in augmentation_config.items():
            console.print(f"  {aug_name}: {aug_params}")
    else:
        console.print(f"[bold]Augmentation:[/bold] None (Baseline)")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        class_to_idx=class_to_idx,
        save_dir=run_dir
    )
    
    # Train model
    console.print("\n[bold]Step 5: Training Model[/bold]")
    trainer.train(args.epochs)
    
    # Test model
    console.print("\n[bold]Step 6: Testing Model[/bold]")
    trainer.test()
    
    # Save code snapshot
    code_dir = run_dir / "code_snapshot"
    code_dir.mkdir(exist_ok=True)
    
    # Copy source files
    for file in ["train_augmentation.py", "trainer.py", "config.py", "dataset.py", "augmentation_config.py"]:
        if Path(file).exists():
            shutil.copy2(file, code_dir)
    
    # Copy models directory
    safe_copytree("models", code_dir / "models")
    
    # Save augmentation config used
    if augmentation_config:
        with open(run_dir / "augmentation_config_used.json", 'w') as f:
            json.dump(augmentation_config, f, indent=2)
    
    # Close wandb run
    wandb.finish()
    
    console.print(f"\n[green]All outputs saved to:[/green] {run_dir}")

if __name__ == "__main__":
    main() 