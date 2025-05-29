"""
Training utilities for Data Balancing Experiments
Extends the original trainer with balancing-specific functionality
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from rich.console import Console
from typing import Dict, Optional, Tuple
from collections import Counter

from config import LOG_INTERVAL, EARLY_STOPPING_PATIENCE
from data_balancing_config import DataBalancingConfig, FocalLoss

console = Console()

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class BalancingTrainer:
    """Training class with balancing-specific functionality"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, test_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: str, class_to_idx: Dict[str, int],
                 save_dir: Path, balancing_technique: Optional[str] = None,
                 loss_function_type: str = "cross_entropy"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.save_dir = save_dir
        self.balancing_technique = balancing_technique
        self.loss_function_type = loss_function_type
        self.is_inception = type(model).__name__ == 'InceptionV3Model'
        
        # Track class distributions for analysis
        self.train_class_counts = self._get_loader_class_counts(train_loader)
        self.val_class_counts = self._get_loader_class_counts(val_loader)
        
        console.print(f"Training class distribution: {self.train_class_counts}")
        console.print(f"Validation class distribution: {self.val_class_counts}")
        
    def _get_loader_class_counts(self, loader: DataLoader) -> Dict[int, int]:
        """Get class distribution from data loader"""
        class_counts = Counter()
        for _, targets in loader:
            for target in targets:
                class_counts[int(target.item())] += 1
        return {int(k): int(v) for k, v in class_counts.items()}
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Handle InceptionV3's auxiliary output during training
                if self.is_inception:
                    output, aux_output = self.model(data)
                    loss1 = self.criterion(output, target)
                    loss2 = self.criterion(aux_output, target)
                    loss = loss1 + 0.4 * loss2  # Auxiliary loss is weighted by 0.4
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
                
                if batch_idx % LOG_INTERVAL == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch_acc": 100. * correct / total,
                        "batch": batch_idx
                    })
        
        return total_loss / len(self.train_loader)
    
    def validate(self, loader: DataLoader, desc: str = "Validation") -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            with tqdm(loader, desc=desc) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)

                    # During validation, we need to enable gradients for InceptionV3's auxiliary output
                    if self.is_inception:
                        # Set model to train mode just for this forward pass to enable auxiliary output
                        self.model.train()
                        output, aux_output = self.model(data)
                        # Switch back to eval mode
                        self.model.eval()
                    else:
                        output = self.model(data)

                    loss = self.criterion(output, target)

                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    
                    # Store predictions for detailed analysis
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

                    pbar.set_postfix({
                        'loss': total_loss / (pbar.n + 1),
                        'acc': 100. * correct / total
                    })

        # Calculate per-class metrics for validation
        val_class_report = classification_report(
            all_targets, all_preds, 
            target_names=[self.idx_to_class[i] for i in range(len(self.class_to_idx))],
            output_dict=True, zero_division=0
        )
        
        # Log per-class metrics if validation
        if desc == "Validation":
            for class_name, metrics in val_class_report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    wandb.log({
                        f"val_class_{class_name}_precision": metrics.get('precision', 0),
                        f"val_class_{class_name}_recall": metrics.get('recall', 0),
                        f"val_class_{class_name}_f1": metrics.get('f1-score', 0)
                    })

        return total_loss / len(loader), 100. * correct / total
    
    def test(self) -> None:
        """Test the model and generate detailed metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device)
                if self.is_inception:
                    # Set model to train mode just for this forward pass
                    self.model.train()
                    output, _ = self.model(data)
                    # Switch back to eval mode
                    self.model.eval()
                else:
                    output = self.model(data)

                # Get probabilities for confidence analysis
                probs = torch.softmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())

                pred = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_targets.extend(target.numpy())
        
        # Generate classification report
        report = classification_report(
            all_targets, all_preds, 
            target_names=[self.idx_to_class[i] for i in range(len(self.class_to_idx))],
            output_dict=True, zero_division=0
        )
        
        # Plot confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=[self.idx_to_class[i] for i in range(len(self.class_to_idx))],
                   yticklabels=[self.idx_to_class[i] for i in range(len(self.class_to_idx))])
        plt.title(f'Confusion Matrix - {self.balancing_technique or "Baseline"}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create per-class analysis
        self._create_per_class_analysis(all_targets, all_preds, all_probs)
        
        # Save metrics
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv(self.save_dir / 'test_metrics.csv')
        
        # Save balancing analysis
        self._save_balancing_analysis(all_targets, all_preds, report)
        
        # Print summary
        console.print("\n[bold]Test Results:[/bold]")
        console.print(f"Balancing Technique: {self.balancing_technique or 'None (Baseline)'}")
        console.print(f"Loss Function: {self.loss_function_type}")
        console.print(metrics_df.round(4).to_string())
        
        # Log final metrics to wandb
        wandb.log({
            "test_accuracy": report['accuracy'],
            "test_f1_macro": report['macro avg']['f1-score'],
            "test_f1_weighted": report['weighted avg']['f1-score'],
            "test_precision_macro": report['macro avg']['precision'],
            "test_recall_macro": report['macro avg']['recall']
        })
    
    def _create_per_class_analysis(self, targets: list, preds: list, probs: np.ndarray):
        """Create detailed per-class analysis"""
        
        # Calculate confidence scores
        max_probs = np.max(probs, axis=1)
        correct_mask = np.array(targets) == np.array(preds)
        
        # Per-class confidence analysis
        class_confidence = {}
        for class_idx in range(len(self.class_to_idx)):
            class_mask = np.array(targets) == class_idx
            if np.sum(class_mask) > 0:
                class_confidence[self.idx_to_class[class_idx]] = {
                    'mean_confidence': np.mean(max_probs[class_mask]),
                    'correct_confidence': np.mean(max_probs[class_mask & correct_mask]) if np.sum(class_mask & correct_mask) > 0 else 0,
                    'incorrect_confidence': np.mean(max_probs[class_mask & ~correct_mask]) if np.sum(class_mask & ~correct_mask) > 0 else 0
                }
        
        # Save confidence analysis
        confidence_df = pd.DataFrame(class_confidence).transpose()
        confidence_df.to_csv(self.save_dir / 'confidence_analysis.csv')
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs[correct_mask], bins=50, alpha=0.7, label='Correct Predictions', density=True)
        plt.hist(max_probs[~correct_mask], bins=50, alpha=0.7, label='Incorrect Predictions', density=True)
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.title(f'Prediction Confidence Distribution - {self.balancing_technique or "Baseline"}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_balancing_analysis(self, targets: list, preds: list, report: dict):
        """Save analysis specific to balancing effectiveness"""
        
        balancing_analysis = {
            "experiment_info": {
                "balancing_technique": self.balancing_technique,
                "loss_function": self.loss_function_type,
                "train_class_distribution": {str(k): int(v) for k, v in self.train_class_counts.items()},
                "val_class_distribution": {str(k): int(v) for k, v in self.val_class_counts.items()}
            },
            "test_performance": {
                "overall_accuracy": report['accuracy'],
                "macro_f1": report['macro avg']['f1-score'],
                "weighted_f1": report['weighted avg']['f1-score']
            },
            "per_class_performance": {}
        }
        
        # Add per-class metrics
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                balancing_analysis["per_class_performance"][class_name] = {
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "f1_score": metrics.get('f1-score', 0),
                    "support": metrics.get('support', 0)
                }
        
        # Calculate imbalance-specific metrics
        test_class_counts = Counter(targets)
        predicted_class_counts = Counter(preds)
        
        balancing_analysis["class_distribution_analysis"] = {
            "test_class_distribution": {str(k): int(v) for k, v in test_class_counts.items()},
            "predicted_class_distribution": {str(k): int(v) for k, v in predicted_class_counts.items()},
            "test_imbalance_ratio": max(test_class_counts.values()) / min(test_class_counts.values()) if test_class_counts else 1
        }
        
        # Save analysis
        import json
        with open(self.save_dir / 'balancing_analysis.json', 'w') as f:
            json.dump(balancing_analysis, f, indent=2)
    
    def train(self, num_epochs: int) -> None:
        """Full training loop"""
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        console.print(f"[bold]Training with {self.balancing_technique or 'baseline'} balancing[/bold]")
        console.print(f"[bold]Loss function: {self.loss_function_type}[/bold]")
        
        for epoch in range(num_epochs):
            console.print(f"\n[bold]Epoch {epoch+1}/{num_epochs}[/bold]")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate(self.val_loader)
            
            # Log metrics
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Save best model based on validation accuracy (better for imbalanced datasets)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'balancing_technique': self.balancing_technique,
                    'loss_function_type': self.loss_function_type,
                    'train_class_counts': self.train_class_counts
                }, self.save_dir / 'best_model.pth')
            
            # Early stopping based on validation loss
            if early_stopping(val_loss):
                console.print("[yellow]Early stopping triggered[/yellow]")
                break
        
        console.print(f"[green]Training completed![/green]")
        console.print(f"[green]Best validation accuracy: {best_val_acc:.2f}%[/green]")
        console.print(f"[green]Best validation loss: {best_val_loss:.4f}[/green]") 