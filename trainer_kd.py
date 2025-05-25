"""
Knowledge Distillation Training utilities and Trainer class
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
from config import LOG_INTERVAL, EARLY_STOPPING_PATIENCE

from kd_model import KnowledgeDistillationModel

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

class KDTrainer:
    """Knowledge Distillation Training class with training and validation loops"""
    
    def __init__(self, kd_model: KnowledgeDistillationModel, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 test_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: str, class_to_idx: Dict[str, int],
                 save_dir: Path):
        self.kd_model = kd_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.save_dir = save_dir
        
    def train_epoch(self) -> Tuple[float, float, float, float]:
        """Train for one epoch with knowledge distillation"""
        self.kd_model.train()
        # Keep teacher models in eval mode
        for teacher in self.kd_model.teacher_models:
            teacher.eval()
        
        total_loss = 0
        total_ce_loss = 0
        total_kd_loss = 0
        total_feature_loss = 0
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with KD
                outputs = self.kd_model(data, target)
                
                loss = outputs['total_loss']
                ce_loss = outputs['ce_loss']
                kd_loss = outputs['kd_loss']
                feature_loss = outputs['feature_loss']
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_kd_loss += kd_loss.item()
                total_feature_loss += feature_loss.item()
                
                pbar.set_postfix({
                    'total': total_loss / (batch_idx + 1),
                    'ce': total_ce_loss / (batch_idx + 1),
                    'kd': total_kd_loss / (batch_idx + 1),
                    'feat': total_feature_loss / (batch_idx + 1)
                })
                
                if batch_idx % LOG_INTERVAL == 0:
                    wandb.log({
                        "batch_total_loss": loss.item(),
                        "batch_ce_loss": ce_loss.item(),
                        "batch_kd_loss": kd_loss.item(),
                        "batch_feature_loss": feature_loss.item(),
                        "batch": batch_idx
                    })
        
        return (total_loss / len(self.train_loader),
                total_ce_loss / len(self.train_loader),
                total_kd_loss / len(self.train_loader),
                total_feature_loss / len(self.train_loader))
    
    def validate(self, loader: DataLoader, desc: str = "Validation") -> Tuple[float, float, float, float, float]:
        """Validate the model with knowledge distillation"""
        self.kd_model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_kd_loss = 0
        total_feature_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(loader, desc=desc) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)

                    outputs = self.kd_model(data, target)
                    
                    loss = outputs['total_loss']
                    ce_loss = outputs['ce_loss']
                    kd_loss = outputs['kd_loss']
                    feature_loss = outputs['feature_loss']
                    student_logits = outputs['student_logits']

                    total_loss += loss.item()
                    total_ce_loss += ce_loss.item()
                    total_kd_loss += kd_loss.item()
                    total_feature_loss += feature_loss.item()
                    
                    pred = student_logits.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

                    pbar.set_postfix({
                        'loss': total_loss / (pbar.n + 1),
                        'acc': 100. * correct / total
                    })

        return (total_loss / len(loader), 
                total_ce_loss / len(loader),
                total_kd_loss / len(loader),
                total_feature_loss / len(loader),
                100. * correct / total)
    
    def test(self) -> None:
        """Test the student model and generate detailed metrics"""
        self.kd_model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device)
                
                outputs = self.kd_model(data)
                student_logits = outputs['student_logits']
                
                pred = student_logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_targets.extend(target.numpy())
        
        # Generate classification report
        report = classification_report(
            all_targets, all_preds, 
            target_names=[self.idx_to_class[i] for i in range(len(self.class_to_idx))],
            output_dict=True
        )
        
        # Plot confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=[self.idx_to_class[i] for i in range(len(self.class_to_idx))],
                   yticklabels=[self.idx_to_class[i] for i in range(len(self.class_to_idx))])
        plt.title('Student Model Confusion Matrix (After Knowledge Distillation)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
        # Save metrics
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv(self.save_dir / 'test_metrics.csv')
        
        # Print summary
        console.print("\n[bold]Student Model Test Results (After KD):[/bold]")
        console.print(metrics_df.round(4).to_string())
    
    def train(self, num_epochs: int) -> None:
        """Full training loop with knowledge distillation"""
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            console.print(f"\n[bold]Epoch {epoch+1}/{num_epochs}[/bold]")
            
            # Train
            train_loss, train_ce, train_kd, train_feat = self.train_epoch()
            
            # Validate
            val_loss, val_ce, val_kd, val_feat, val_acc = self.validate(self.val_loader)
            
            # Log metrics
            wandb.log({
                "train_total_loss": train_loss,
                "train_ce_loss": train_ce,
                "train_kd_loss": train_kd,
                "train_feature_loss": train_feat,
                "val_total_loss": val_loss,
                "val_ce_loss": val_ce,
                "val_kd_loss": val_kd,
                "val_feature_loss": val_feat,
                "val_acc": val_acc,
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            # Print epoch summary (more concise like original trainer)
            console.print(f"Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, KD: {train_kd:.4f}, Feat: {train_feat:.4f})")
            console.print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save both the KD model and the student model separately
                torch.save({
                    'epoch': epoch,
                    'kd_model_state_dict': self.kd_model.state_dict(),
                    'student_model_state_dict': self.kd_model.get_student_model().state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'kd_config': self.kd_model.get_config()
                }, self.save_dir / 'best_model.pth')
                
                # Also save student model separately for easy loading
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.kd_model.get_student_model().state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / 'best_student_model.pth')
            
            # Early stopping
            if early_stopping(val_loss):
                console.print("[yellow]Early stopping triggered[/yellow]")
                break
        
        console.print("[green]Knowledge Distillation training completed![/green]") 