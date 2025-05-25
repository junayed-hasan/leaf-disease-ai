"""
Main Knowledge Distillation Training Script
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
from pathlib import Path
from config import *
from dataset import load_data, analyze_dataset, visualize_samples
from models import ModelFactory
from trainer_kd import KDTrainer
from kd_model import KnowledgeDistillationModel

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

def get_kd_run_dir(student_model: str, experiment_name: str) -> Path:
    """Get the output directory for KD run"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = BASE_OUTPUT_DIR / "knowledge_distillation" / student_model / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def find_teacher_model_paths(teacher_models: list) -> list:
    """Find the best model checkpoints for teacher models"""
    teacher_paths = []
    outputs_dir = Path("outputs")
    
    for teacher_model in teacher_models:
        # Look for the model directory
        model_dir = outputs_dir / teacher_model
        if not model_dir.exists():
            raise FileNotFoundError(f"Teacher model directory not found: {model_dir}")
        
        # Find the combined dataset model specifically
        best_model_path = None
        for subdir in model_dir.iterdir():
            if subdir.is_dir() and "combined_" in subdir.name:
                potential_path = subdir / "best_model.pth"
                if potential_path.exists():
                    best_model_path = potential_path
                    break
        
        if best_model_path is None:
            raise FileNotFoundError(f"No combined dataset best_model.pth found for teacher {teacher_model}")
        
        teacher_paths.append(str(best_model_path))
        console.print(f"Found teacher {teacher_model}: {best_model_path}")
    
    return teacher_paths

def find_student_model_path(student_model: str) -> str:
    """Find the pretrained student model checkpoint"""
    outputs_dir = Path("outputs")
    model_dir = outputs_dir / student_model
    
    if not model_dir.exists():
        console.print(f"[yellow]Warning: Student model directory not found: {model_dir}[/yellow]")
        return None
    
    # Find the combined dataset model specifically
    for subdir in model_dir.iterdir():
        if subdir.is_dir() and "combined_" in subdir.name:
            potential_path = subdir / "best_model.pth"
            if potential_path.exists():
                console.print(f"Found pretrained student {student_model}: {potential_path}")
                return str(potential_path)
    
    console.print(f"[yellow]Warning: No combined dataset pretrained weights found for student {student_model}[/yellow]")
    return None

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    
    # Model arguments
    parser.add_argument('--student_model', type=str, required=True,
                      choices=ModelFactory.list_available_models(),
                      help='Student model architecture')
    parser.add_argument('--teacher_models', type=str, nargs='+', 
                      default=['densenet121', 'resnet101', 'densenet201', 'efficientnet_b4'],
                      help='Teacher model architectures (default: best ensemble)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='combined',
                      choices=['plantvillage', 'tomatovillage', 'combined'],
                      help='Dataset to use for training')
    
    # KD-specific arguments
    parser.add_argument('--alpha', type=float, default=1/3,
                      help='Weight for classification loss (default: 1/3)')
    parser.add_argument('--beta', type=float, default=1/3,
                      help='Weight for logit distillation loss (default: 1/3)')
    parser.add_argument('--gamma', type=float, default=1/3,
                      help='Weight for feature distillation loss (default: 1/3)')
    
    # Temperature arguments
    parser.add_argument('--temperature', type=float, default=4.0,
                      help='Temperature for basic KD (default: 4.0)')
    parser.add_argument('--teacher_temp', type=float, default=None,
                      help='Teacher temperature for advanced scaling (optional)')
    parser.add_argument('--student_temp', type=float, default=None,
                      help='Student temperature for advanced scaling (optional)')
    
    # Feature distillation
    parser.add_argument('--use_feature_distillation', action='store_true',
                      help='Enable feature-level distillation')
    
    # Experiment name
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for the experiment')
    
    args = parser.parse_args()
    
    # Validate temperature scaling arguments
    if (args.teacher_temp is None) != (args.student_temp is None):
        raise ValueError("Both teacher_temp and student_temp must be specified together or not at all")
    
    dataset_config = get_dataset_config(args.dataset)
    
    # Create experiment name
    if args.experiment_name is None:
        temp_info = f"T{args.temperature}" if args.teacher_temp is None else f"Tt{args.teacher_temp}_Ts{args.student_temp}"
        feat_info = "_feat" if args.use_feature_distillation else ""
        teacher_info = "_".join(args.teacher_models)
        args.experiment_name = f"kd_{teacher_info}_to_{args.student_model}_{temp_info}{feat_info}"
    
    # Create run directory
    run_dir = get_kd_run_dir(args.student_model, args.experiment_name)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Set seed
    set_seed(SEED)
    
    # Find teacher model paths
    teacher_paths = find_teacher_model_paths(args.teacher_models)
    
    # Find student model path (optional)
    student_path = find_student_model_path(args.student_model)
    
    # Get model-specific image size (use student model's requirements)
    image_size = get_image_size(args.student_model)
    
    # Create config dictionary
    config = {
        "student_model": args.student_model,
        "teacher_models": args.teacher_models,
        "dataset": dataset_config['name'],
        "dataset_type": dataset_config['type'],
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "image_size": image_size,
        "pretrained": PRETRAINED,
        "experiment_name": args.experiment_name,
        "output_dir": str(run_dir),
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "temperature": args.temperature,
        "teacher_temp": args.teacher_temp,
        "student_temp": args.student_temp,
        "use_feature_distillation": args.use_feature_distillation,
        "teacher_paths": teacher_paths,
        "student_path": student_path
    }
    
    # Save config
    save_config(config, run_dir)
    
    # Initialize wandb
    run = wandb.init(
        project="tomato-disease-kd",
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
    train_loader, val_loader, test_loader, class_to_idx = load_data(args.dataset, args.student_model)
    
    # Save class mapping
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    
    # Visualize samples
    console.print("\n[bold]Step 3: Visualizing Samples[/bold]")
    visualize_samples(args.dataset, class_to_idx, run_dir)
    
    # Initialize KD model
    console.print("\n[bold]Step 4: Initializing Knowledge Distillation Model[/bold]")
    kd_model = KnowledgeDistillationModel(
        teacher_models=args.teacher_models,
        teacher_paths=teacher_paths,
        student_model_name=args.student_model,
        student_path=student_path,
        num_classes=dataset_config['num_classes'],
        device=DEVICE,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
        use_feature_distillation=args.use_feature_distillation
    ).to(DEVICE)
    
    # Only optimize student model parameters
    optimizer = Adam(kd_model.student_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, 
                                 factor=SCHEDULER_FACTOR, min_lr=MIN_LR)
    
    # Print model information
    console.print(f"[bold]Teacher Models:[/bold] {', '.join(args.teacher_models)}")
    console.print(f"[bold]Student Model:[/bold] {args.student_model}")
    console.print(f"[bold]KD Configuration:[/bold]")
    console.print(f"  α (CE Loss): {args.alpha}")
    console.print(f"  β (KD Loss): {args.beta}")
    console.print(f"  γ (Feature Loss): {args.gamma}")
    if args.teacher_temp is not None:
        console.print(f"  Teacher Temperature: {args.teacher_temp}")
        console.print(f"  Student Temperature: {args.student_temp}")
    else:
        console.print(f"  Temperature: {args.temperature}")
    console.print(f"  Feature Distillation: {args.use_feature_distillation}")
    
    # Initialize trainer
    trainer = KDTrainer(
        kd_model=kd_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        class_to_idx=class_to_idx,
        save_dir=run_dir
    )
    
    # Train model
    console.print("\n[bold]Step 5: Training with Knowledge Distillation[/bold]")
    trainer.train(NUM_EPOCHS)
    
    # Test model
    console.print("\n[bold]Step 6: Testing Student Model[/bold]")
    trainer.test()
    
    # Save code snapshot
    code_dir = run_dir / "code_snapshot"
    code_dir.mkdir(exist_ok=True)
    
    # Copy source files
    for file in ["train_kd.py", "trainer_kd.py", "kd_model.py", "config.py", "dataset.py"]:
        shutil.copy2(file, code_dir)
    
    # Copy models directory
    safe_copytree("models", code_dir / "models")
    
    # Close wandb run
    wandb.finish()
    
    console.print(f"\n[green]All outputs saved to:[/green] {run_dir}")

if __name__ == "__main__":
    main() 