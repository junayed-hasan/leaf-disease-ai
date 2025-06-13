#!/usr/bin/env python3
"""
Evaluate Best Ensemble Model on Test Datasets
Tests the trained ensemble model on test sets of tomatoVillage and plantVillage datasets
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
from models import ModelFactory
from test_dataset_loader import (
    create_test_dataloader, 
    analyze_test_dataset, 
    visualize_test_samples,
    get_available_test_datasets
)

console = Console()

class EnsembleTestEvaluator:
    """Ensemble model evaluator for test datasets"""
    
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
        all_samples = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, samples) in enumerate(data_loader):
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
                all_samples.extend(samples)
                
                if batch_idx % 50 == 0:
                    console.print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs), all_samples

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

def load_original_class_mapping():
    """Load the original combined dataset class mapping"""
    
    # Try to find a recent class mapping file
    class_mapping_files = list(Path("outputs").rglob("class_mapping.json"))
    
    # Look for a combined dataset class mapping
    for mapping_file in class_mapping_files:
        if "combined" in str(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
    
    # If no combined mapping found, use the first available
    if class_mapping_files:
        console.print("[yellow]Using first available class mapping file[/yellow]")
        with open(class_mapping_files[0], 'r') as f:
            return json.load(f)
    
    # Fallback to default mapping
    console.print("[red]No class mapping file found, using default[/red]")
    return {
        "Tomato___Bacterial_spot": 0,
        "Tomato___Early_blight": 1,
        "Tomato___Late_blight": 2,
        "Tomato___Leaf_Miner": 3,
        "Tomato___Leaf_Mold": 4,
        "Tomato___Magnesium_Deficiency": 5,
        "Tomato___Nitrogen_Deficiency": 6,
        "Tomato___Pottassium_Deficiency": 7,
        "Tomato___Septoria_leaf_spot": 8,
        "Tomato___Spider_mites Two-spotted_spider_mite": 9,
        "Tomato___Spotted_Wilt_Virus": 10,
        "Tomato___Target_Spot": 11,
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 12,
        "Tomato___Tomato_mosaic_virus": 13,
        "Tomato___healthy": 14
    }

def evaluate_ensemble_on_test_dataset(ensemble_preds, ensemble_targets, ensemble_probs, 
                                     all_samples, class_mapping, save_dir, dataset_name):
    """Evaluate ensemble performance on test dataset"""
    
    # Create class name mappings
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
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
        "dataset": dataset_name,
        "total_samples": len(ensemble_targets),
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
    unique_targets = np.unique(ensemble_targets)
    for target_idx in unique_targets:
        if target_idx < len(precision):
            class_name = idx_to_class[target_idx]
            performance["per_class_metrics"][class_name] = {
                "precision": precision[target_idx] * 100,
                "recall": recall[target_idx] * 100,
                "f1_score": f1[target_idx] * 100,
                "support": int(support[target_idx])
            }
    
    # Print results
    console.print(f"\n[bold]Ensemble Performance on {dataset_name} Test Dataset:[/bold]")
    console.print(f"Total Samples: {len(ensemble_targets)}")
    console.print(f"Accuracy: {accuracy*100:.2f}%")
    console.print(f"Precision (Macro): {precision_macro*100:.2f}%")
    console.print(f"Recall (Macro): {recall_macro*100:.2f}%")
    console.print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
    console.print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
    
    # Print per-class results
    console.print(f"\n[bold]Per-Class Results:[/bold]")
    for target_idx in sorted(unique_targets):
        if target_idx < len(precision):
            class_name = idx_to_class[target_idx]
            console.print(f"  {class_name}: F1={f1[target_idx]*100:.2f}%, Support={int(support[target_idx])}")
    
    # Save detailed classification report
    report = classification_report(
        ensemble_targets, ensemble_preds, 
        target_names=[idx_to_class[i] for i in sorted(unique_targets)],
        output_dict=True, 
        zero_division=0
    )
    
    with open(save_dir / f"{dataset_name}_ensemble_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Create and save confusion matrix
    cm = confusion_matrix(ensemble_targets, ensemble_preds)
    
    plt.figure(figsize=(12, 10))
    class_labels = [idx_to_class[i] for i in sorted(unique_targets)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Ensemble Performance on {dataset_name} Test Dataset - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / f"{dataset_name}_ensemble_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze misclassifications
    misclassified = []
    for i, (pred, target, sample) in enumerate(zip(ensemble_preds, ensemble_targets, all_samples)):
        if pred != target:
            misclassified.append({
                'sample_idx': i,
                'predicted_class': idx_to_class[pred],
                'actual_class': idx_to_class[target],
                'original_class_name': sample['original_class_name'],
                'confidence': float(np.max(ensemble_probs[i]))
            })
    
    # Save misclassification analysis
    with open(save_dir / f"{dataset_name}_ensemble_misclassifications.json", "w") as f:
        json.dump(misclassified[:100], f, indent=4)  # Save top 100 misclassifications
    
    console.print(f"Misclassified samples: {len(misclassified)}/{len(ensemble_targets)} ({len(misclassified)/len(ensemble_targets)*100:.1f}%)")
    
    # Save performance metrics
    with open(save_dir / f"{dataset_name}_ensemble_performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=4)
    
    # Log to wandb
    wandb.log({
        f"{dataset_name}_ensemble_accuracy": accuracy * 100,
        f"{dataset_name}_ensemble_f1_macro": f1_macro * 100,
        f"{dataset_name}_ensemble_f1_weighted": f1_weighted * 100,
        f"{dataset_name}_ensemble_precision_macro": precision_macro * 100,
        f"{dataset_name}_ensemble_recall_macro": recall_macro * 100,
        f"{dataset_name}_ensemble_total_samples": len(ensemble_targets),
        f"{dataset_name}_ensemble_misclassification_rate": len(misclassified)/len(ensemble_targets)*100
    })
    
    return performance

def evaluate_single_dataset(dataset_name: str, ensemble_evaluator, class_mapping, save_dir):
    """Evaluate ensemble on a single test dataset"""
    
    console.print(f"\n[bold]Evaluating on {dataset_name} Test Dataset[/bold]")
    
    # Analyze test dataset
    console.print(f"\n[bold]Analyzing {dataset_name} Test Dataset[/bold]")
    dataset_stats = analyze_test_dataset(dataset_name, class_mapping, save_dir)
    
    if dataset_stats['total_test_samples'] == 0:
        console.print(f"[yellow]No test samples found for {dataset_name}, skipping[/yellow]")
        return None
    
    # Visualize test samples
    console.print(f"\n[bold]Visualizing {dataset_name} Test Samples[/bold]")
    visualize_test_samples(dataset_name, class_mapping, save_dir)
    
    # Create test dataloader
    console.print(f"\n[bold]Loading {dataset_name} Test Data[/bold]")
    image_size = 224  # Use standard image size
    test_loader, test_dataset = create_test_dataloader(
        dataset_name, class_mapping, image_size=image_size, batch_size=32
    )
    
    # Get ensemble predictions
    console.print(f"\n[bold]Getting Ensemble Predictions on {dataset_name}[/bold]")
    ensemble_preds, ensemble_targets, ensemble_probs, all_samples = ensemble_evaluator.predict(test_loader)
    
    # Evaluate ensemble
    console.print(f"\n[bold]Evaluating Ensemble Performance on {dataset_name}[/bold]")
    performance = evaluate_ensemble_on_test_dataset(
        ensemble_preds, ensemble_targets, ensemble_probs, all_samples,
        class_mapping, save_dir, dataset_name
    )
    
    return {
        "dataset_stats": dataset_stats,
        "performance": performance
    }

def main():
    """Evaluate the best ensemble model on test datasets"""
    
    console.print("[bold]Evaluating Best Ensemble Model on Test Datasets[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get available test datasets
    available_datasets = get_available_test_datasets()
    console.print(f"[bold]Available Test Datasets:[/bold] {', '.join(available_datasets)}")
    
    if not available_datasets:
        console.print("[red]No test datasets found. Please ensure test directories exist.[/red]")
        return False
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = Path("outputs") / "test_evaluation" / "ensemble" / f"ensemble_on_test_datasets_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Load original class mapping
    console.print("\n[bold]Step 1: Loading Original Class Mapping[/bold]")
    class_mapping = load_original_class_mapping()
    console.print(f"Dataset classes: {len(class_mapping)}")
    
    # Find trained models
    console.print("\n[bold]Step 2: Finding Trained Ensemble Models[/bold]")
    model_paths = find_best_ensemble_models()
    
    if len(model_paths) == 0:
        console.print("[red]No trained ensemble models found. Please run train_best_ensemble.py first[/red]")
        return False
    
    console.print(f"Found {len(model_paths)} trained models")
    
    # Create config
    config = {
        "evaluation_type": "ensemble_on_test_datasets",
        "ensemble_models": list(model_paths.keys()),
        "test_datasets": available_datasets,
        "total_classes": len(class_mapping),
        "model_paths": {k: str(v) for k, v in model_paths.items()},
        "output_dir": str(run_dir)
    }
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize wandb
    wandb_name = f"ensemble_test_eval_{timestamp}"
    run = wandb.init(
        project="tomato-disease-test-evaluation",
        name=wandb_name,
        config=config
    )
    
    try:
        # Create ensemble model
        console.print("\n[bold]Step 3: Creating Ensemble Model[/bold]")
        ensemble_evaluator = EnsembleTestEvaluator(
            model_names=list(model_paths.keys()),
            model_paths=list(model_paths.values()),
            num_classes=len(class_mapping),
            device=DEVICE
        )
        
        # Evaluate on each test dataset
        results = {}
        for dataset_name in available_datasets:
            console.print(f"\n[bold]Step 4.{len(results)+1}: Evaluating on {dataset_name}[/bold]")
            result = evaluate_single_dataset(dataset_name, ensemble_evaluator, class_mapping, run_dir)
            if result is not None:
                results[dataset_name] = result
        
        # Save overall evaluation summary
        evaluation_summary = {
            "evaluation_type": "ensemble_on_test_datasets",
            "ensemble_models": list(model_paths.keys()),
            "evaluated_datasets": list(results.keys()),
            "results": results,
            "evaluation_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "test_evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4)
        
        # Print summary
        console.print(f"\n[bold]Ensemble Test Evaluation Summary:[/bold]")
        for dataset_name, result in results.items():
            performance = result["performance"]
            console.print(f"  {dataset_name}:")
            console.print(f"    F1 Score (Macro): {performance['f1_macro']:.2f}%")
            console.print(f"    Accuracy: {performance['accuracy']:.2f}%")
            console.print(f"    Test Samples: {performance['total_samples']}")
        
        console.print(f"\n[green]Ensemble evaluation on test datasets completed successfully![/green]")
        console.print(f"Results saved to: {run_dir}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Ensemble test evaluation failed with exception: {e}[/red]")
        raise e
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 