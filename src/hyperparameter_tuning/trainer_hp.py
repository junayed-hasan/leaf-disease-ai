"""
Training utilities and Trainer class
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

class Trainer:
    """Training class with training and validation loops"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, test_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: str, class_to_idx: Dict[str, int],
                 save_dir: Path):
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
        self.is_inception = type(model).__name__ == 'InceptionV3Model'
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
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
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                
                if batch_idx % LOG_INTERVAL == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": batch_idx
                    })
        
        return total_loss / len(self.train_loader)
    
    def validate(self, loader: DataLoader, desc: str = "Validation") -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

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

                    pbar.set_postfix({
                        'loss': total_loss / (pbar.n + 1),
                        'acc': 100. * correct / total
                    })

        return total_loss / len(loader), 100. * correct / total
    
    def test(self) -> None:
        """Test the model and generate detailed metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []

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

                pred = output.argmax(dim=1).cpu().numpy()
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
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
        # Save metrics
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv(self.save_dir / 'test_metrics.csv')
        
        # Print summary
        console.print("\n[bold]Test Results:[/bold]")
        console.print(metrics_df.round(4).to_string())
    
    def train(self, num_epochs: int) -> None:
        """Full training loop"""
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
        best_val_loss = float('inf')
        
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
                "epoch": epoch
            })
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / 'best_model.pth')
            
            # Early stopping
            if early_stopping(val_loss):
                console.print("[yellow]Early stopping triggered[/yellow]")
                break
        
        console.print("[green]Training completed![/green]")