"""
Ensemble model creation and testing script for tomato leaf disease classification
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import itertools
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from rich.console import Console
from rich.table import Table
import json
import argparse
from datetime import datetime
import os
import shutil

from config import *
from dataset import load_data, analyze_dataset, visualize_samples
from models import ModelFactory

console = Console()

class EnsembleModel:
    """
    Ensemble model that combines multiple pre-trained models
    """
    def __init__(self, models, model_paths, model_names, ensemble_type="hard", weights=None, device="cuda"):
        """
        Initialize the ensemble model
        
        Args:
            models: List of model objects
            model_paths: List of paths to model weights
            model_names: List of model names for reference
            ensemble_type: Type of ensemble ('hard', 'weighted_hard', or 'soft')
            weights: Optional weights for weighted voting (must match length of models)
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.models = models
        self.model_paths = model_paths
        self.model_names = model_names
        self.ensemble_type = ensemble_type
        self.weights = weights if weights is not None else [1] * len(models)
        self.device = device
        
        # Load model weights
        for i, model in enumerate(self.models):
            checkpoint = torch.load(self.model_paths[i], map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
        if ensemble_type == "weighted_hard" and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, data_loader):
        """
        Make predictions using the ensemble
        
        Args:
            data_loader: DataLoader containing test data
            
        Returns:
            all_preds: List of predicted class indices
            all_targets: List of true class indices
            all_probs: List of probability arrays (only for soft voting)
        """
        all_model_preds = [[] for _ in range(len(self.models))]
        all_model_probs = [[] for _ in range(len(self.models))]
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                all_targets.extend(target.numpy())
                
                # Get predictions from each model
                for i, model in enumerate(self.models):
                    if "InceptionV3" in self.model_names[i]:
                        model.train()  # Set to train mode for aux output
                        output, _ = model(data)
                        model.eval()  # Set back to eval mode
                    else:
                        output = model(data)
                    
                    # Store probabilities
                    probs = torch.softmax(output, dim=1).cpu().numpy()
                    all_model_probs[i].extend(probs)
                    
                    # Store hard predictions
                    preds = output.argmax(dim=1).cpu().numpy()
                    all_model_preds[i].extend(preds)
        
        # Combine predictions based on ensemble type
        if self.ensemble_type == "hard":
            # Simple majority vote
            all_preds = []
            for i in range(len(all_targets)):
                votes = [all_model_preds[j][i] for j in range(len(self.models))]
                pred = max(set(votes), key=votes.count)
                all_preds.append(pred)
            
            return all_preds, all_targets, None
            
        elif self.ensemble_type == "weighted_hard":
            # Weighted majority vote
            all_preds = []
            for i in range(len(all_targets)):
                # Count weighted votes for each class
                vote_counts = {}
                for j in range(len(self.models)):
                    pred = all_model_preds[j][i]
                    vote_counts[pred] = vote_counts.get(pred, 0) + self.weights[j]
                
                # Get class with highest weighted vote
                pred = max(vote_counts.items(), key=lambda x: x[1])[0]
                all_preds.append(pred)
            
            return all_preds, all_targets, None
            
        elif self.ensemble_type == "soft":
            # Average probabilities
            all_probs = []
            all_preds = []
            
            for i in range(len(all_targets)):
                # Average the probabilities from each model
                avg_prob = np.zeros_like(all_model_probs[0][i])
                for j in range(len(self.models)):
                    avg_prob += all_model_probs[j][i] * self.weights[j]
                avg_prob /= sum(self.weights)
                
                all_probs.append(avg_prob)
                all_preds.append(np.argmax(avg_prob))
            
            return all_preds, all_targets, all_probs
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def get_ensemble_info(self):
        """Return information about the ensemble"""
        return {
            "model_names": self.model_names,
            "ensemble_type": self.ensemble_type,
            "weights": self.weights
        }

def evaluate_ensemble(all_preds, all_targets, all_probs, idx_to_class, save_dir):
    """
    Evaluate ensemble performance and save metrics
    
    Args:
        all_preds: List of predicted class indices
        all_targets: List of true class indices
        all_probs: List of probability arrays (only for soft voting)
        idx_to_class: Mapping from class indices to class names
        save_dir: Directory to save results
    """
    # Generate classification report
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Create performance summary
    performance = {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100
    }
    
    # Create DataFrame from report
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(save_dir / 'test_metrics.csv')
    
    # Save performance summary
    with open(save_dir / 'performance_summary.json', 'w') as f:
        json.dump(performance, f, indent=4)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()
    
    # Print summary
    console.print("\n[bold]Test Results:[/bold]")
    console.print(f"Accuracy: {accuracy * 100:.2f}%")
    console.print(f"Precision: {precision * 100:.2f}%")
    console.print(f"Recall: {recall * 100:.2f}%")
    console.print(f"F1 Score: {f1 * 100:.2f}%")
    
    return performance

def load_model(model_name, num_classes, model_path, device):
    """Load a model with its weights"""
    model = ModelFactory.get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False  # We'll load our own weights
    ).to(device)
    
    return model

def get_combination_name(model_names):
    """Get a short name for the model combination"""
    return "+".join([name.split("_")[-1] for name in model_names])

def create_ensemble_dir(combination, ensemble_type, dataset_name):
    """Create directory for ensemble results"""
    combo_name = get_combination_name(combination)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    dir_name = f"ensemble_{ensemble_type}_{combo_name}_{dataset_name}_{timestamp}"
    ensemble_dir = Path("outputs") / "ensembles" / dir_name
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    return ensemble_dir

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create and test ensemble models')
    parser.add_argument('--dataset', type=str, default='combined',
                     choices=['plantvillage', 'tomatovillage', 'combined'],
                     help='Dataset to use for evaluation')
    parser.add_argument('--ensemble_types', type=str, nargs='+', 
                     default=['hard', 'weighted_hard', 'soft'],
                     help='Ensemble methods to use')
    parser.add_argument('--models', type=str, nargs='+',
                     default=['densenet121', 'resnet101', 'densenet201', 'efficientnet_b4', 'efficient_vit'],
                     help='Models to include in ensembles')
    parser.add_argument('--min_models', type=int, default=3,
                     help='Minimum number of models in an ensemble')
    parser.add_argument('--max_models', type=int, default=5,
                     help='Maximum number of models in an ensemble')
    args = parser.parse_args()
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    num_classes = dataset_config['num_classes']
    
    # Load test data
    _, _, test_loader, class_to_idx = load_data(args.dataset)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Create results directory
    results_dir = Path("outputs") / "ensembles"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary file for all ensembles
    summary_file = results_dir / f"ensemble_summary_{args.dataset}.csv"
    summary_data = []
    
    # Test all combinations
    total_combinations = 0
    for k in range(args.min_models, args.max_models + 1):
        total_combinations += len(list(itertools.combinations(args.models, k)))
    
    console.print(f"[bold]Testing {total_combinations} model combinations with {len(args.ensemble_types)} ensemble types[/bold]")
    console.print(f"Total experiments: {total_combinations * len(args.ensemble_types)}")
    
    # Track the best ensemble
    best_ensemble = {
        "f1_score": 0,
        "combination": None,
        "ensemble_type": None,
        "save_dir": None
    }
    
    # Process each model size (3, 4, 5 models)
    for k in range(args.min_models, args.max_models + 1):
        console.print(f"\n[bold]Testing combinations with {k} models[/bold]")
        
        # Generate all combinations of k models
        combinations = list(itertools.combinations(args.models, k))
        
        # Test each combination
        for combo_idx, combination in enumerate(combinations):
            console.print(f"\nCombination {combo_idx+1}/{len(combinations)}: {', '.join(combination)}")
            
            # Find model paths for each model in this combination
            model_paths = []
            for model_name in combination:
                # Use the combined dataset version of the model
                model_dir = Path("outputs") / model_name
                combined_dir = None
                
                # Find the combined dataset folder
                for subdir in model_dir.iterdir():
                    if subdir.is_dir() and "combined" in subdir.name:
                        combined_dir = subdir
                        break
                        
                if combined_dir is None:
                    console.print(f"[red]Could not find combined dataset model for {model_name}[/red]")
                    continue
                    
                model_path = combined_dir / "best_model.pth"
                model_paths.append(model_path)
            
            # Skip if any model is missing
            if len(model_paths) != len(combination):
                continue
                
            # Load models for this combination
            models = []
            for i, model_name in enumerate(combination):
                model = load_model(model_name, num_classes, model_paths[i], DEVICE)
                models.append(model)
            
            # Test each ensemble type
            for ensemble_type in args.ensemble_types:
                console.print(f"  Testing {ensemble_type} ensemble...")
                
                # Create directory for this ensemble
                save_dir = create_ensemble_dir(combination, ensemble_type, args.dataset)
                
                # Create ensemble weights (equal for hard and soft, F1 scores for weighted_hard)
                weights = None
                if ensemble_type == "weighted_hard" or ensemble_type == "soft":
                    # Use F1 scores from baseline as weights
                    weights = []
                    model_f1_scores = {
                        "densenet121": 94.00,
                        "resnet101": 93.35,
                        "densenet201": 91.57,
                        "efficientnet_b4": 91.65,
                        "efficient_vit": 91.84
                    }
                    for model_name in combination:
                        weights.append(model_f1_scores.get(model_name, 1.0))
                
                # Create and evaluate ensemble
                ensemble = EnsembleModel(
                    models=models,
                    model_paths=model_paths,
                    model_names=combination,
                    ensemble_type=ensemble_type,
                    weights=weights,
                    device=DEVICE
                )
                
                # Get predictions
                all_preds, all_targets, all_probs = ensemble.predict(test_loader)
                
                # Evaluate ensemble
                performance = evaluate_ensemble(
                    all_preds, all_targets, all_probs, 
                    idx_to_class, save_dir
                )
                
                # Save ensemble info
                ensemble_info = ensemble.get_ensemble_info()
                ensemble_info.update({
                    "performance": performance,
                    "num_models": len(combination)
                })
                
                with open(save_dir / "ensemble_info.json", "w") as f:
                    json.dump(ensemble_info, f, indent=4)
                
                # Add to summary
                summary_entry = {
                    "ensemble_type": ensemble_type,
                    "model_combination": "+".join(combination),
                    "num_models": len(combination),
                    "accuracy": performance["accuracy"],
                    "precision": performance["precision"],
                    "recall": performance["recall"],
                    "f1_score": performance["f1_score"],
                    "save_dir": str(save_dir)
                }
                summary_data.append(summary_entry)
                
                # Update best ensemble if needed
                if performance["f1_score"] > best_ensemble["f1_score"]:
                    best_ensemble = {
                        "f1_score": performance["f1_score"],
                        "combination": combination,
                        "ensemble_type": ensemble_type,
                        "save_dir": save_dir,
                        "performance": performance
                    }
    
    # Save summary data
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    # Print best ensemble
    console.print("\n[bold green]Best Ensemble:[/bold green]")
    console.print(f"Model Combination: {', '.join(best_ensemble['combination'])}")
    console.print(f"Ensemble Type: {best_ensemble['ensemble_type']}")
    console.print(f"F1 Score: {best_ensemble['f1_score']:.2f}%")
    console.print(f"Results saved to: {best_ensemble['save_dir']}")
    
    # Create a symlink to the best ensemble
    best_link = results_dir / f"best_ensemble_{args.dataset}"
    if os.path.exists(best_link):
        os.remove(best_link)
    os.symlink(best_ensemble['save_dir'], best_link, target_is_directory=True)
    
    console.print(f"\nAll ensemble results saved to: {results_dir}")
    console.print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 