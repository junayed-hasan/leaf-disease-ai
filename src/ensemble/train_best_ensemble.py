#!/usr/bin/env python3
"""
Train Best Ensemble Model
Uses the optimal combination: densenet121, resnet101, densenet201, efficientnet_b4
With best augmentations, best hyperparameters, and best balancing technique (ADASYN)
Follows the same pipeline flow as train.py and train_balancing.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import wandb
from rich.console import Console
from datetime import datetime
import random
import numpy as np
import shutil
import json
import os
from pathlib import Path

# Import existing modules
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
    """Safely copy directory tree even if destination exists"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def train_single_model(model_name: str, dataset_name: str = "combined") -> Path:
    """Train a single model with best configurations"""
    
    console.print(f"\n[bold blue]Training {model_name} with best configurations[/bold blue]")
    
    # Load best augmentation config
    augmentation_config_path = Path("combined_augmentation_1.json")
    augmentation_config = None
    if augmentation_config_path.exists():
        with open(augmentation_config_path, 'r') as f:
            augmentation_config = json.load(f)
        console.print(f"[bold]Using best augmentation config from:[/bold] {augmentation_config_path}")
    else:
        console.print("[yellow]Warning: Best augmentation config not found, using baseline[/yellow]")
    
    dataset_config = get_dataset_config(dataset_name)
    
    # Use best hyperparameters
    best_hp = DataBalancingConfig.BEST_HYPERPARAMETERS
    console.print(f"[bold]Using best hyperparameters:[/bold] {best_hp}")
    
    # Create experiment name
    experiment_name = f"best_ensemble_{model_name}"
    
    # Create run directory
    run_dir = get_run_dir(model_name, experiment_name)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Set seed
    set_seed(SEED)
    
    # Get model-specific image size
    image_size = get_image_size(model_name)
    
    # Create config dictionary
    config = {
        "model": model_name,
        "dataset": dataset_config['name'],
        "dataset_type": dataset_config['type'],
        "epochs": 50,
        "batch_size": 16,  # Reduced batch size for CUDA memory
        "balancing_technique": "adasyn",
        "loss_function": "cross_entropy",
        "image_size": image_size,
        "pretrained": PRETRAINED,
        "experiment_name": experiment_name,
        "output_dir": str(run_dir),
        "augmentation_config": augmentation_config,
        "best_hyperparameters": best_hp,
        "ensemble_training": True
    }
    
    # Save config
    save_config(config, run_dir)
    
    # Create meaningful run name for wandb
    wandb_name = f"best_ensemble_{dataset_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb
    run = wandb.init(
        project="tomato-disease-best-ensemble",
        name=wandb_name,
        config=config
    )
    
    try:
        # Analyze dataset
        console.print("[bold]Step 1: Analyzing Dataset[/bold]")
        stats = analyze_balanced_dataset(dataset_name, "adasyn", run_dir)
        
        # Save dataset statistics
        with open(run_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        # Load data with balancing technique
        console.print("\n[bold]Step 2: Loading Data with Balancing[/bold]")
        train_loader, val_loader, test_loader, class_to_idx = load_balanced_data(
            dataset_name, model_name, 
            balancing_technique="adasyn",
            augmentation_config=augmentation_config,
            batch_size=16
        )
        
        # Save class mapping
        with open(run_dir / "class_mapping.json", "w") as f:
            json.dump(class_to_idx, f, indent=4)
        
        # Visualize samples
        console.print("\n[bold]Step 3: Visualizing Samples[/bold]")
        visualize_samples(dataset_name, class_to_idx, run_dir)
        
        # Create model
        console.print("\n[bold]Step 4: Creating Model[/bold]")
        model = ModelFactory.get_model(
            model_name, 
            dataset_config['num_classes'], 
            pretrained=PRETRAINED
        ).to(DEVICE)
        
        # Initialize criterion
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
        console.print(f"[bold]Model:[/bold] {model_name}")
        console.print(f"[bold]Dataset:[/bold] {dataset_config['name']} ({dataset_config['num_classes']} classes)")
        console.print(f"[bold]Image Size:[/bold] {image_size}x{image_size}")
        console.print(f"[bold]Training Configuration:[/bold]")
        console.print(f"  Epochs: 50")
        console.print(f"  Batch Size: 16 (reduced for CUDA memory)")
        console.print(f"  Balancing Technique: ADASYN")
        console.print(f"  Loss Function: cross_entropy")
        
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
            balancing_technique="adasyn",
            loss_function_type="cross_entropy"
        )
        
        # Train model
        console.print("\n[bold]Step 5: Training Model[/bold]")
        trainer.train(50)
        
        # Test model
        console.print("\n[bold]Step 6: Testing Model[/bold]")
        trainer.test()
        
        # Save code snapshot
        code_dir = run_dir / "code_snapshot"
        code_dir.mkdir(exist_ok=True)
        
        # Copy source files
        for file in ["train_best_ensemble.py", "trainer_balancing.py", "config.py", 
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
            "experiment_type": "best_ensemble_training",
            "model": model_name,
            "dataset": dataset_name,
            "balancing_technique": "adasyn",
            "loss_function": "cross_entropy",
            "best_hyperparameters": best_hp,
            "training_completed": True
        }
        
        with open(run_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        console.print(f"[green]✓ {model_name} training completed successfully[/green]")
        
        return run_dir
        
    except Exception as e:
        console.print(f"[red]✗ {model_name} training failed with exception: {e}[/red]")
        raise e
    finally:
        # Close wandb run
        wandb.finish()

def main():
    """Train the best ensemble models with optimal configurations"""
    
    console.print("[bold]Training Best Ensemble Models[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Best ensemble combination - TEMPORARILY ONLY EFFICIENTNET_B4
    best_models = ['efficientnet_b4']  # Only train efficientnet_b4 due to CUDA memory
    
    # Best configurations
    best_hyperparameters = DataBalancingConfig.BEST_HYPERPARAMETERS
    
    console.print(f"[bold]Best Ensemble Models:[/bold] {', '.join(best_models)}")
    console.print(f"[bold]Best Balancing Technique:[/bold] ADASYN")
    console.print(f"[bold]Best Augmentation:[/bold] combined_augmentation_1.json")
    console.print(f"[bold]Best Hyperparameters:[/bold] {best_hyperparameters}")
    
    results = {}
    trained_model_paths = {}
    
    # Train each model in the best ensemble with optimal configurations
    for model_name in best_models:
        try:
            model_run_dir = train_single_model(model_name)
            results[model_name] = "success"
            trained_model_paths[model_name] = model_run_dir
        except Exception as e:
            console.print(f"[red]✗ {model_name} training failed: {e}[/red]")
            results[model_name] = "failed"
    
    # Print summary
    console.print(f"\n[bold]Best Ensemble Training Summary:[/bold]")
    console.print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = sum(1 for status in results.values() if status == "success")
    total = len(results)
    
    console.print(f"Successful: {successful}/{total}")
    
    if successful < total:
        console.print("Failed models:")
        for model_name, status in results.items():
            if status != "success":
                console.print(f"  - {model_name}: {status}")
    
    # Save ensemble training summary
    ensemble_summary = {
        "best_ensemble_models": best_models,
        "training_results": results,
        "trained_model_paths": {k: str(v) for k, v in trained_model_paths.items()},
        "best_hyperparameters": best_hyperparameters,
        "training_completed_at": datetime.now().isoformat()
    }
    
    summary_dir = Path("outputs") / "ensemble_training_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_dir / "best_ensemble_training_summary.json", 'w') as f:
        json.dump(ensemble_summary, f, indent=2)
    
    if successful == total:
        console.print("[green]All best ensemble models trained successfully![/green]")
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("1. Run ensemble evaluation: python evaluate_best_ensemble.py")
        console.print("2. Run knowledge distillation: python train_best_kd.py")
    else:
        console.print(f"[yellow]{total - successful} model(s) failed[/yellow]")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 