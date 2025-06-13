#!/usr/bin/env python3
"""
Train Best Knowledge Distillation Model
Uses the best teacher ensemble (densenet121, resnet101, densenet201, efficientnet_b4)
to distill knowledge to the best student model (shufflenet_v2_x0_5)
With optimal KD configurations: Temperature=15, alpha=0.7, beta=0.3
Follows the same pipeline flow as train_kd.py but uses ensemble as teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from dataset_balancing import load_balanced_data, analyze_balanced_dataset
from dataset import visualize_samples
from models import ModelFactory
from trainer_balancing import BalancingTrainer
from data_balancing_config import DataBalancingConfig

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

class EnsembleTeacher:
    """Ensemble teacher model for knowledge distillation"""
    
    def __init__(self, model_names, model_paths, num_classes, device):
        self.model_names = model_names
        self.model_paths = model_paths
        self.num_classes = num_classes
        self.device = device
        self.models = []
        
        # Load all teacher models
        for model_name, model_path in zip(model_names, model_paths):
            console.print(f"Loading teacher {model_name} from {model_path}")
            model = self._load_model(model_name, model_path)
            model.eval()  # Set to evaluation mode
            # Freeze teacher models
            for param in model.parameters():
                param.requires_grad = False
            self.models.append(model)
        
        console.print(f"[green]Successfully loaded {len(self.models)} teacher models[/green]")
    
    def _load_model(self, model_name, model_path):
        """Load a single teacher model"""
        model = ModelFactory.get_model(model_name, self.num_classes, pretrained=False).to(self.device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def forward(self, x):
        """Get ensemble teacher outputs using soft voting"""
        with torch.no_grad():
            # Get predictions from all teacher models
            model_outputs = []
            for model in self.models:
                outputs = model(x)
                model_outputs.append(outputs)
            
            # Average logits (soft voting)
            ensemble_outputs = torch.stack(model_outputs).mean(dim=0)
            
        return ensemble_outputs

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation Loss with optimal parameters"""
    
    def __init__(self, alpha=0.7, beta=0.3, temperature=15):
        super().__init__()
        self.alpha = alpha  # Weight for hard loss (student vs ground truth)
        self.beta = beta    # Weight for soft loss (student vs teacher)
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs, teacher_outputs, targets):
        # Hard loss (student vs ground truth)
        hard_loss = self.ce_loss(student_outputs, targets)
        
        # Soft loss (student vs teacher)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + self.beta * soft_loss
        
        return total_loss, hard_loss, soft_loss

class KDTrainer:
    """Knowledge Distillation Trainer"""
    
    def __init__(self, student_model, teacher_ensemble, train_loader, val_loader, test_loader,
                 optimizer, scheduler, device, class_to_idx, save_dir, kd_loss):
        self.student_model = student_model
        self.teacher_ensemble = teacher_ensemble
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_to_idx = class_to_idx
        self.save_dir = save_dir
        self.kd_loss = kd_loss
        
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.student_model.train()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get teacher outputs (frozen)
            teacher_outputs = self.teacher_ensemble.forward(inputs)
            
            # Get student outputs
            student_outputs = self.student_model(inputs)
            
            # Calculate KD loss
            loss, hard_loss, soft_loss = self.kd_loss(student_outputs, teacher_outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            
            if batch_idx % 50 == 0:
                console.print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                            f"Loss: {loss.item():.4f}, Hard: {hard_loss.item():.4f}, Soft: {soft_loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_hard_loss = total_hard_loss / len(self.train_loader)
        avg_soft_loss = total_soft_loss / len(self.train_loader)
        
        return avg_loss, avg_hard_loss, avg_soft_loss
    
    def validate(self):
        """Validate the student model"""
        self.student_model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get teacher and student outputs
                teacher_outputs = self.teacher_ensemble.forward(inputs)
                student_outputs = self.student_model(inputs)
                
                # Calculate loss
                loss, _, _ = self.kd_loss(student_outputs, teacher_outputs, targets)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(student_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import f1_score, accuracy_score
        accuracy = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, f1_macro, f1_weighted
    
    def train(self, num_epochs):
        """Train the student model with knowledge distillation"""
        console.print(f"[bold]Starting Knowledge Distillation Training for {num_epochs} epochs[/bold]")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_hard_loss, train_soft_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_accuracy, val_f1_macro, val_f1_weighted = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_hard_loss": train_hard_loss,
                "train_soft_loss": train_soft_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy * 100,
                "val_f1_macro": val_f1_macro * 100,
                "val_f1_weighted": val_f1_weighted * 100,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_f1_macro > self.best_val_f1:
                self.best_val_f1 = val_f1_macro
                self.save_checkpoint(epoch, is_best=True)
                console.print(f"[green]New best F1 score: {val_f1_macro*100:.2f}%[/green]")
            
            # Print epoch summary
            console.print(f"Epoch {epoch}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Val F1: {val_f1_macro*100:.2f}%, Val Acc: {val_accuracy*100:.2f}%")
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1_macro)
        
        console.print(f"[bold]Training completed! Best F1 Score: {self.best_val_f1*100:.2f}%[/bold]")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'class_to_idx': self.class_to_idx
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pth")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pth")
    
    def test(self):
        """Test the best student model"""
        console.print("[bold]Testing best student model...[/bold]")
        
        # Load best model
        checkpoint = torch.load(self.save_dir / "best_model.pth", map_location=self.device)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.student_model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.student_model(inputs)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import classification_report, f1_score, accuracy_score
        accuracy = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        
        # Print results
        console.print(f"[bold]Test Results:[/bold]")
        console.print(f"Accuracy: {accuracy*100:.2f}%")
        console.print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
        console.print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
        
        # Log to wandb
        wandb.log({
            "test_accuracy": accuracy * 100,
            "test_f1_macro": f1_macro * 100,
            "test_f1_weighted": f1_weighted * 100
        })
        
        # Save detailed results
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(self.class_to_idx))]
        
        report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
        with open(self.save_dir / "test_classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        return accuracy, f1_macro, f1_weighted

def find_best_ensemble_models():
    """Find the trained ensemble models"""
    
    # Check if we have the ensemble training summary
    summary_path = Path("outputs") / "ensemble_training_summary" / "best_ensemble_training_summary.json"
    
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        model_paths = {}
        for model_name, run_dir in summary["trained_model_paths"].items():
            model_path = Path(run_dir) / "best_model.pth"
            if model_path.exists():
                model_paths[model_name] = model_path
            else:
                console.print(f"[yellow]Warning: Model file not found for {model_name} at {model_path}[/yellow]")
        
        return model_paths
    
    # Fallback: search for models manually
    console.print("[yellow]Ensemble training summary not found, searching for models manually[/yellow]")
    
    best_models = ['densenet121', 'resnet101', 'densenet201', 'efficientnet_b4']
    model_paths = {}
    
    for model_name in best_models:
        model_dir = Path("outputs") / model_name
        
        # Find the most recent best_ensemble run
        best_ensemble_dirs = [d for d in model_dir.iterdir() 
                             if d.is_dir() and "best_ensemble" in d.name]
        
        if best_ensemble_dirs:
            # Get the most recent one
            latest_dir = max(best_ensemble_dirs, key=lambda x: x.stat().st_mtime)
            model_path = latest_dir / "best_model.pth"
            
            if model_path.exists():
                model_paths[model_name] = model_path
                console.print(f"Found {model_name} at {model_path}")
            else:
                console.print(f"[red]Model file not found for {model_name} at {model_path}[/red]")
        else:
            console.print(f"[red]No best_ensemble directory found for {model_name}[/red]")
    
    return model_paths

def main():
    """Train the best knowledge distillation model"""
    
    console.print("[bold]Training Best Knowledge Distillation Model[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Best configurations
    best_teacher_models = ['densenet121', 'resnet101', 'densenet201', 'efficientnet_b4']
    best_student_model = 'shufflenet_v2_x0_5'
    dataset_name = "combined"
    
    # Best KD hyperparameters
    best_kd_config = {
        "temperature": 15,
        "alpha": 0.7,  # Student's hard loss weight
        "beta": 0.3,   # Distillation loss weight (KL divergence)
    }
    
    console.print(f"[bold]Best Teacher Models:[/bold] {', '.join(best_teacher_models)}")
    console.print(f"[bold]Best Student Model:[/bold] {best_student_model}")
    console.print(f"[bold]Best KD Configuration:[/bold] {best_kd_config}")
    console.print(f"[bold]Best Balancing Technique:[/bold] ADASYN")
    console.print(f"[bold]Best Augmentation:[/bold] combined_augmentation_1.json")
    
    # Find teacher models
    console.print("\n[bold]Step 1: Finding Teacher Models[/bold]")
    teacher_model_paths = find_best_ensemble_models()
    
    if len(teacher_model_paths) != len(best_teacher_models):
        missing_models = set(best_teacher_models) - set(teacher_model_paths.keys())
        console.print(f"[red]Missing teacher models: {missing_models}[/red]")
        console.print("[red]Please run train_best_ensemble.py first[/red]")
        return False
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = Path("outputs") / "knowledge_distillation" / best_student_model / f"best_kd_distillation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Set seed
    set_seed(SEED)
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    num_classes = dataset_config['num_classes']
    
    # Load best augmentation config
    augmentation_config_path = Path("combined_augmentation_1.json")
    augmentation_config = None
    if augmentation_config_path.exists():
        with open(augmentation_config_path, 'r') as f:
            augmentation_config = json.load(f)
        console.print(f"[bold]Using best augmentation config from:[/bold] {augmentation_config_path}")
    else:
        console.print("[yellow]Warning: Best augmentation config not found, using baseline[/yellow]")
    
    # Use best hyperparameters
    best_hp = DataBalancingConfig.BEST_HYPERPARAMETERS
    
    # Get model-specific image size (use student model's requirements)
    image_size = get_image_size(best_student_model)
    
    # Create config dictionary
    config = {
        "student_model": best_student_model,
        "teacher_models": best_teacher_models,
        "dataset": dataset_config['name'],
        "dataset_type": dataset_config['type'],
        "epochs": 50,
        "batch_size": 32,
        "balancing_technique": "adasyn",
        "image_size": image_size,
        "pretrained": PRETRAINED,
        "experiment_name": "best_kd_distillation",
        "output_dir": str(run_dir),
        "augmentation_config": augmentation_config,
        "best_hyperparameters": best_hp,
        "kd_config": best_kd_config,
        "teacher_model_paths": {k: str(v) for k, v in teacher_model_paths.items()}
    }
    
    # Save config
    save_config(config, run_dir)
    
    # Initialize wandb
    wandb_name = f"best_kd_{dataset_name}_{best_student_model}_{timestamp}"
    run = wandb.init(
        project="tomato-disease-best-kd",
        name=wandb_name,
        config=config
    )
    
    try:
        # Analyze dataset
        console.print("\n[bold]Step 2: Analyzing Dataset[/bold]")
        stats = analyze_balanced_dataset(dataset_name, "adasyn", run_dir)
        
        # Save dataset statistics
        with open(run_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        # Load data with balancing technique
        console.print("\n[bold]Step 3: Loading Data with Balancing[/bold]")
        train_loader, val_loader, test_loader, class_to_idx = load_balanced_data(
            dataset_name, best_student_model, 
            balancing_technique="adasyn",
            augmentation_config=augmentation_config,
            batch_size=32
        )
        
        # Save class mapping
        with open(run_dir / "class_mapping.json", "w") as f:
            json.dump(class_to_idx, f, indent=4)
        
        # Visualize samples
        console.print("\n[bold]Step 4: Visualizing Samples[/bold]")
        visualize_samples(dataset_name, class_to_idx, run_dir)
        
        # Create teacher ensemble
        console.print("\n[bold]Step 5: Creating Teacher Ensemble[/bold]")
        teacher_ensemble = EnsembleTeacher(
            model_names=list(teacher_model_paths.keys()),
            model_paths=list(teacher_model_paths.values()),
            num_classes=num_classes,
            device=DEVICE
        )
        
        # Create student model
        console.print("\n[bold]Step 6: Creating Student Model[/bold]")
        student_model = ModelFactory.get_model(
            best_student_model, 
            num_classes, 
            pretrained=PRETRAINED
        ).to(DEVICE)
        
        # Create KD loss
        kd_loss = KnowledgeDistillationLoss(
            alpha=best_kd_config["alpha"],
            beta=best_kd_config["beta"],
            temperature=best_kd_config["temperature"]
        )
        
        # Initialize optimizer and scheduler with best hyperparameters
        optimizer = Adam(
            student_model.parameters(), 
            lr=best_hp["learning_rate"], 
            weight_decay=best_hp["weight_decay"]
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, 
                                     factor=SCHEDULER_FACTOR, min_lr=MIN_LR)
        
        # Print model information
        console.print(f"[bold]Teacher Models:[/bold] {', '.join(best_teacher_models)}")
        console.print(f"[bold]Student Model:[/bold] {best_student_model}")
        console.print(f"[bold]KD Configuration:[/bold]")
        console.print(f"  α (Hard Loss): {best_kd_config['alpha']}")
        console.print(f"  β (Soft Loss): {best_kd_config['beta']}")
        console.print(f"  Temperature: {best_kd_config['temperature']}")
        
        # Initialize trainer
        trainer = KDTrainer(
            student_model=student_model,
            teacher_ensemble=teacher_ensemble,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            class_to_idx=class_to_idx,
            save_dir=run_dir,
            kd_loss=kd_loss
        )
        
        # Train model
        console.print("\n[bold]Step 7: Training with Knowledge Distillation[/bold]")
        trainer.train(50)
        
        # Test model
        console.print("\n[bold]Step 8: Testing Student Model[/bold]")
        test_accuracy, test_f1_macro, test_f1_weighted = trainer.test()
        
        # Save code snapshot
        code_dir = run_dir / "code_snapshot"
        code_dir.mkdir(exist_ok=True)
        
        # Copy source files
        for file in ["train_best_kd.py", "config.py", "dataset_balancing.py", "data_balancing_config.py"]:
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
            "experiment_type": "best_knowledge_distillation",
            "student_model": best_student_model,
            "teacher_models": best_teacher_models,
            "dataset": dataset_name,
            "balancing_technique": "adasyn",
            "best_hyperparameters": best_hp,
            "kd_config": best_kd_config,
            "final_test_metrics": {
                "accuracy": test_accuracy * 100,
                "f1_macro": test_f1_macro * 100,
                "f1_weighted": test_f1_weighted * 100
            },
            "training_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        console.print(f"\n[green]Knowledge distillation training completed successfully![/green]")
        console.print(f"Final Test F1 Score: {test_f1_macro*100:.2f}%")
        console.print(f"Results saved to: {run_dir}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Knowledge distillation training failed with exception: {e}[/red]")
        raise e
    finally:
        # Close wandb run
        wandb.finish()
    
    console.print(f"\n[bold]Best Knowledge Distillation Training Summary:[/bold]")
    console.print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 