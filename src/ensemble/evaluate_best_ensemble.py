#!/usr/bin/env python3
"""
Evaluate Best Ensemble Model
Uses the optimal combination: densenet121, resnet101, densenet201, efficientnet_b4
With soft voting for best performance
Follows the same pipeline flow as ensemble.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.console import Console
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Import existing modules
from config import *
from dataset import load_data, analyze_dataset, visualize_samples
from models import ModelFactory

console = Console()

class BestEnsembleModel:
    """Best ensemble model using soft voting"""
    
    def __init__(self, model_names, model_paths, num_classes, device):
        self.model_names = model_names
        self.model_paths = model_paths
        self.num_classes = num_classes
        self.device = device
        self.models = []
        
        # Load all models
        for i, (model_name, model_path) in enumerate(zip(model_names, model_paths)):
            console.print(f"Loading {model_name} from {model_path}")
            model = self._load_model(model_name, model_path)
            model.eval()  # Set to evaluation mode
            self.models.append(model)
        
        console.print(f"[green]Successfully loaded {len(self.models)} models for ensemble[/green]")
    
    def _load_model(self, model_name, model_path):
        """Load a single model"""
        model = ModelFactory.get_model(model_name, self.num_classes, pretrained=False).to(self.device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def predict(self, data_loader):
        """Get ensemble predictions using soft voting"""
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get predictions from all models
                model_probs = []
                for model in self.models:
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    model_probs.append(probs)
                
                # Average probabilities (soft voting)
                ensemble_probs = torch.stack(model_probs).mean(dim=0)
                ensemble_preds = torch.argmax(ensemble_probs, dim=1)
                
                all_preds.extend(ensemble_preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(ensemble_probs.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    console.print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def find_best_ensemble_models():
    """Find the trained models from the ensemble training"""
    
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

def evaluate_ensemble(ensemble_preds, ensemble_targets, ensemble_probs, class_to_idx, save_dir):
    """Evaluate ensemble performance"""
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    # Calculate metrics
    accuracy = accuracy_score(ensemble_targets, ensemble_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        ensemble_targets, ensemble_preds, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        ensemble_targets, ensemble_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        ensemble_targets, ensemble_preds, average='weighted', zero_division=0
    )
    
    # Create performance dictionary
    performance = {
        "accuracy": accuracy * 100,
        "precision_macro": precision_macro * 100,
        "recall_macro": recall_macro * 100,
        "f1_macro": f1_macro * 100,
        "precision_weighted": precision_weighted * 100,
        "recall_weighted": recall_weighted * 100,
        "f1_weighted": f1_weighted * 100,
        "per_class_metrics": {}
    }
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        performance["per_class_metrics"][class_name] = {
            "precision": precision[i] * 100,
            "recall": recall[i] * 100,
            "f1_score": f1[i] * 100,
            "support": int(support[i])
        }
    
    # Print results
    console.print(f"\n[bold]Best Ensemble Performance:[/bold]")
    console.print(f"Accuracy: {accuracy*100:.2f}%")
    console.print(f"Precision (Macro): {precision_macro*100:.2f}%")
    console.print(f"Recall (Macro): {recall_macro*100:.2f}%")
    console.print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
    console.print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
    
    # Save detailed classification report
    report = classification_report(
        ensemble_targets, ensemble_preds, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    with open(save_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Create and save confusion matrix
    cm = confusion_matrix(ensemble_targets, ensemble_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Best Ensemble - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save performance metrics
    with open(save_dir / "performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=4)
    
    # Log to wandb
    wandb.log({
        "ensemble_accuracy": accuracy * 100,
        "ensemble_f1_macro": f1_macro * 100,
        "ensemble_f1_weighted": f1_weighted * 100,
        "ensemble_precision_macro": precision_macro * 100,
        "ensemble_recall_macro": recall_macro * 100
    })
    
    return performance

def main():
    """Evaluate the best ensemble model"""
    
    console.print("[bold]Evaluating Best Ensemble Model[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Best ensemble combination
    best_models = ['densenet121', 'resnet101', 'densenet201', 'efficientnet_b4']
    dataset_name = "combined"
    
    console.print(f"[bold]Best Ensemble Models:[/bold] {', '.join(best_models)}")
    console.print(f"[bold]Ensemble Type:[/bold] Soft Voting")
    console.print(f"[bold]Dataset:[/bold] {dataset_name}")
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = Path("outputs") / "ensemble" / f"best_ensemble_evaluation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Find trained models
    console.print("\n[bold]Step 1: Finding Trained Models[/bold]")
    model_paths = find_best_ensemble_models()
    
    if len(model_paths) != len(best_models):
        missing_models = set(best_models) - set(model_paths.keys())
        console.print(f"[red]Missing models: {missing_models}[/red]")
        console.print("[red]Please run train_best_ensemble.py first[/red]")
        return False
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    num_classes = dataset_config['num_classes']
    
    # Create config
    config = {
        "ensemble_models": best_models,
        "ensemble_type": "soft_voting",
        "dataset": dataset_config['name'],
        "dataset_type": dataset_config['type'],
        "num_classes": num_classes,
        "model_paths": {k: str(v) for k, v in model_paths.items()},
        "experiment_name": "best_ensemble_evaluation",
        "output_dir": str(run_dir)
    }
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize wandb
    wandb_name = f"best_ensemble_eval_{dataset_name}_{timestamp}"
    run = wandb.init(
        project="tomato-disease-best-ensemble",
        name=wandb_name,
        config=config
    )
    
    try:
        # Analyze dataset
        console.print("\n[bold]Step 2: Analyzing Dataset[/bold]")
        stats, image_sizes = analyze_dataset(dataset_name, run_dir)
        
        # Save dataset statistics
        with open(run_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        # Load data (use the first model's image size for consistency)
        console.print("\n[bold]Step 3: Loading Data[/bold]")
        _, _, test_loader, class_to_idx = load_data(dataset_name, best_models[0])
        
        # Save class mapping
        with open(run_dir / "class_mapping.json", "w") as f:
            json.dump(class_to_idx, f, indent=4)
        
        # Visualize samples
        console.print("\n[bold]Step 4: Visualizing Samples[/bold]")
        visualize_samples(dataset_name, class_to_idx, run_dir)
        
        # Create ensemble model
        console.print("\n[bold]Step 5: Creating Ensemble Model[/bold]")
        ensemble_model = BestEnsembleModel(
            model_names=list(model_paths.keys()),
            model_paths=list(model_paths.values()),
            num_classes=num_classes,
            device=DEVICE
        )
        
        # Get ensemble predictions
        console.print("\n[bold]Step 6: Getting Ensemble Predictions[/bold]")
        ensemble_preds, ensemble_targets, ensemble_probs = ensemble_model.predict(test_loader)
        
        # Evaluate ensemble
        console.print("\n[bold]Step 7: Evaluating Ensemble[/bold]")
        performance = evaluate_ensemble(
            ensemble_preds, ensemble_targets, ensemble_probs, 
            class_to_idx, run_dir
        )
        
        # Save ensemble info
        ensemble_info = {
            "model_names": list(model_paths.keys()),
            "model_paths": {k: str(v) for k, v in model_paths.items()},
            "ensemble_type": "soft_voting",
            "num_models": len(model_paths),
            "performance": performance,
            "evaluation_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "ensemble_info.json", "w") as f:
            json.dump(ensemble_info, f, indent=4)
        
        console.print(f"\n[green]Best ensemble evaluation completed successfully![/green]")
        console.print(f"F1 Score (Macro): {performance['f1_macro']:.2f}%")
        console.print(f"Results saved to: {run_dir}")
        
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("1. Run knowledge distillation: python train_best_kd.py")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Best ensemble evaluation failed with exception: {e}[/red]")
        raise e
    finally:
        # Close wandb run
        wandb.finish()
    
    console.print(f"\n[bold]Best Ensemble Evaluation Summary:[/bold]")
    console.print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 