"""
Main training script
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from rich.console import Console
import argparse
from datetime import datetime
import random
import numpy as np
import shutil
import json
import os
from config import *
from dataset import load_data, analyze_dataset, visualize_samples
from models import ModelFactory
from trainer import Trainer

console = Console()

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(config: dict, save_dir: Path) -> None:
    """Save experiment configuration"""
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def safe_copytree(src: str, dst: Path) -> None:
    """Safely copy directory tree even if destination exists"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train leaf disease classifier')
    parser.add_argument('--model', type=str, default='resnet50',
                      choices=ModelFactory.list_available_models(),
                      help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='plantvillage',
                      choices=['plantvillage', 'tomatovillage', 'combined'],  # Added 'combined' option
                      help='Dataset to use for training')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for the experiment')
    args = parser.parse_args()
    
    dataset_config = get_dataset_config(args.dataset)
    
    if args.experiment_name is None:
        args.experiment_name = f"baseline_{args.dataset}_{args.model}"
    
    # Create run directory
    run_dir = get_run_dir(args.model, args.dataset, args.experiment_name)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Set seed
    set_seed(SEED)
    
    # Get model-specific image size
    image_size = get_image_size(args.model)
    
    # Create config dictionary
    config = {
        "architecture": args.model,
        "dataset": dataset_config['name'],
        "dataset_type": dataset_config['type'],
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "image_size": image_size,
        "pretrained": PRETRAINED,
        "experiment_name": args.experiment_name,
        "output_dir": str(run_dir)
    }
    
    # Save config
    save_config(config, run_dir)
    
    # Initialize wandb
    run = wandb.init(
        project="tomato-disease-classification",
        name=f"{args.dataset}_{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config
    )
    
    # Analyze dataset
    console.print("[bold]Step 1: Analyzing Dataset[/bold]")
    stats, image_sizes = analyze_dataset(args.dataset, run_dir)
    
    # Save dataset statistics
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    # Load data
    console.print("\n[bold]Step 2: Loading Data[/bold]")
    train_loader, val_loader, test_loader, class_to_idx = load_data(args.dataset, args.model)
    
    # Save class mapping
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    
    # Visualize samples
    console.print("\n[bold]Step 3: Visualizing Samples[/bold]")
    visualize_samples(args.dataset, class_to_idx, run_dir)
    
    # Initialize model
    console.print("\n[bold]Step 4: Initializing Model[/bold]")
    model = ModelFactory.get_model(
        model_name=args.model,
        num_classes=dataset_config['num_classes'],
        pretrained=PRETRAINED
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, 
                                 factor=SCHEDULER_FACTOR, min_lr=MIN_LR)
    
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
    trainer.train(NUM_EPOCHS)
    
    # Test model
    console.print("\n[bold]Step 6: Testing Model[/bold]")
    trainer.test()
    
    # Save code snapshot
    code_dir = run_dir / "code_snapshot"
    code_dir.mkdir(exist_ok=True)
    
    # Copy source files
    for file in ["train.py", "config.py", "dataset.py", "trainer.py"]:
        shutil.copy2(file, code_dir)
    
    # Copy models directory
    safe_copytree("models", code_dir / "models")
    
    # Close wandb run
    wandb.finish()
    
    console.print(f"\n[green]All outputs saved to:[/green] {run_dir}")

if __name__ == "__main__":
    main()