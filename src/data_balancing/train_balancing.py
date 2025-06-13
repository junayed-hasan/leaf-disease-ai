#!/usr/bin/env python3
"""
Training Script for Data Balancing Experiments
Based on original train.py but with data balancing techniques
Uses best hyperparameters and augmentation configuration
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import wandb
from rich.console import Console

from config import *
from dataset import visualize_samples, get_dataset_config
from dataset_balancing import load_balanced_data, analyze_balanced_dataset
from models import ModelFactory
from trainer_balancing import BalancingTrainer
from data_balancing_config import DataBalancingConfig
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
    parser = argparse.ArgumentParser(description='Train model with data balancing techniques')
    
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
    
    # Data balancing arguments
    parser.add_argument('--balancing_technique', type=str, default=None,
                      choices=['random_oversampling', 'smote', 'adasyn', 'offline_augmentation', 'focal_loss'],
                      help='Data balancing technique to apply')
    parser.add_argument('--loss_function', type=str, default='cross_entropy',
                      choices=['cross_entropy', 'focal_loss'],
                      help='Loss function to use')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for the experiment')
    
    args = parser.parse_args()
    
    # Load the best augmentation configuration
    augmentation_config = None
    augmentation_config_path = "combined_augmentation_1.json"
    if Path(augmentation_config_path).exists():
        with open(augmentation_config_path, 'r') as f:
            augmentation_config = json.load(f)
        console.print(f"[bold]Using best augmentation config from:[/bold] {augmentation_config_path}")
    else:
        console.print("[yellow]Warning: Best augmentation config not found, using baseline[/yellow]")
    
    dataset_config = get_dataset_config(args.dataset)
    
    # Use best hyperparameters
    best_hp = DataBalancingConfig.BEST_HYPERPARAMETERS
    console.print(f"[bold]Using best hyperparameters:[/bold] {best_hp}")
    
    # Create experiment name
    if args.experiment_name is None:
        balancing_desc = args.balancing_technique or "baseline"
        loss_desc = args.loss_function
        args.experiment_name = f"{args.model}_balance_{balancing_desc}_{loss_desc}"
    
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
        "balancing_technique": args.balancing_technique,
        "loss_function": args.loss_function,
        "image_size": image_size,
        "pretrained": PRETRAINED,
        "experiment_name": args.experiment_name,
        "output_dir": str(run_dir),
        "augmentation_config": augmentation_config,
        "best_hyperparameters": best_hp
    }
    
    # Save config
    save_config(config, run_dir)
    
    # Create meaningful run name for wandb
    balancing_description = f"{args.balancing_technique or 'baseline'}_{args.loss_function}"
    wandb_name = f"{args.dataset}_{args.model}_balance_{balancing_description}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb
    run = wandb.init(
        project="tomato-disease-balancing",
        name=wandb_name,
        config=config
    )
    
    # Analyze dataset
    console.print("[bold]Step 1: Analyzing Dataset[/bold]")
    stats = analyze_balanced_dataset(args.dataset, args.balancing_technique, run_dir)
    
    # Save dataset statistics
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    # Load data with balancing technique
    console.print("\n[bold]Step 2: Loading Data with Balancing[/bold]")
    train_loader, val_loader, test_loader, class_to_idx = load_balanced_data(
        args.dataset, args.model, 
        balancing_technique=args.balancing_technique,
        augmentation_config=augmentation_config,
        batch_size=args.batch_size
    )
    
    # Save class mapping
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    
    # Visualize samples (using original dataset for consistency)
    console.print("\n[bold]Step 3: Visualizing Samples[/bold]")
    visualize_samples(args.dataset, class_to_idx, run_dir)
    
    # Create model
    console.print("\n[bold]Step 4: Creating Model[/bold]")
    model = ModelFactory.get_model(
        args.model, 
        dataset_config['num_classes'], 
        pretrained=PRETRAINED
    ).to(DEVICE)
    
    # Initialize criterion based on loss function
    if args.loss_function == "focal_loss":
        # Get class counts for focal loss alpha computation
        sample_batch = next(iter(train_loader))
        _, sample_targets = sample_batch
        class_counts = {}
        for _, targets in train_loader:
            for target in targets:
                class_idx = target.item()
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        criterion = DataBalancingConfig.get_focal_loss(class_counts, DEVICE)
        console.print(f"[bold]Using Focal Loss with computed alpha weights[/bold]")
    else:
        criterion = nn.CrossEntropyLoss()
        console.print(f"[bold]Using Cross Entropy Loss[/bold]")
    
    # Initialize optimizer and scheduler with best hyperparameters
    optimizer = Adam(
        model.parameters(), 
        lr=best_hp["learning_rate"], 
        weight_decay=best_hp["weight_decay"]
    )
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)  # Step scheduler
    
    # Print experiment information
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print(f"[bold]Dataset:[/bold] {dataset_config['name']} ({dataset_config['num_classes']} classes)")
    console.print(f"[bold]Image Size:[/bold] {image_size}x{image_size}")
    console.print(f"[bold]Training Configuration:[/bold]")
    console.print(f"  Epochs: {args.epochs}")
    console.print(f"  Batch Size: {args.batch_size}")
    console.print(f"  Balancing Technique: {args.balancing_technique or 'None (Baseline)'}")
    console.print(f"  Loss Function: {args.loss_function}")
    
    console.print(f"[bold]Best Hyperparameters:[/bold]")
    for hp_name, hp_value in best_hp.items():
        console.print(f"  {hp_name}: {hp_value}")
    
    if augmentation_config:
        console.print(f"[bold]Augmentation Configuration (Best):[/bold]")
        for aug_name, aug_params in augmentation_config.items():
            console.print(f"  {aug_name}: {aug_params}")
    else:
        console.print(f"[bold]Augmentation:[/bold] None (Baseline)")
    
    # Initialize trainer
    trainer = BalancingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        class_to_idx=class_to_idx,
        save_dir=run_dir,
        balancing_technique=args.balancing_technique,
        loss_function_type=args.loss_function
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
    for file in ["train_balancing.py", "trainer_balancing.py", "config.py", 
                 "dataset_balancing.py", "data_balancing_config.py"]:
        if Path(file).exists():
            shutil.copy2(file, code_dir)
    
    # Copy models directory
    safe_copytree("models", code_dir / "models")
    
    # Save augmentation config used
    if augmentation_config:
        with open(run_dir / "augmentation_config_used.json", 'w') as f:
            json.dump(augmentation_config, f, indent=2)
    
    # Save final summary
    summary = {
        "experiment_type": "data_balancing",
        "balancing_technique": args.balancing_technique,
        "loss_function": args.loss_function,
        "model": args.model,
        "dataset": args.dataset,
        "best_hyperparameters": best_hp,
        "final_metrics": {
            # These will be filled by wandb logs
            "test_accuracy": None,
            "test_f1_macro": None,
            "test_f1_weighted": None
        }
    }
    
    with open(run_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Close wandb run
    wandb.finish()
    
    console.print(f"\n[green]All outputs saved to:[/green] {run_dir}")

if __name__ == "__main__":
    main() 