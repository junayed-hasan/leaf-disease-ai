#!/usr/bin/env python3
"""
Evaluate ONNX Quantized Model on Individual Test Datasets
Tests the ONNX quantized model on TomatoVillage and PlantVillage test datasets individually
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Logging will be disabled.")
    WANDB_AVAILABLE = False
    wandb = None

try:
    from rich.console import Console
    console = Console()
except ImportError:
    print("Warning: rich not available. Using basic print statements.")
    class SimpleConsole:
        def print(self, msg, **kwargs):
            print(msg)
    console = SimpleConsole()

# Import existing modules with error handling
try:
    from config import *
except ImportError as e:
    print(f"Warning: Could not import config: {e}")

try:
    from onnx_model_evaluator import ONNXModelEvaluator, ONNXModelAnalyzer
except ImportError as e:
    print(f"Error: Could not import ONNX evaluator: {e}")
    print("Make sure onnx_model_evaluator.py is in the same directory")
    sys.exit(1)

try:
    from test_dataset_loader import TestDatasetLoader
except ImportError as e:
    print(f"Error: Could not import TestDatasetLoader: {e}")
    print("Make sure test_dataset_loader.py is available")
    sys.exit(1)

try:
    from dataset import visualize_samples
except ImportError:
    def visualize_samples(*args, **kwargs):
        print("Visualization skipped - dataset module not available")

class ONNXModelTestEvaluator:
    """ONNX quantized model evaluator for individual test datasets"""
    
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.device = 'cpu'
        
        # Load the ONNX quantized model
        console.print(f"Loading ONNX quantized model from {onnx_model_path}")
        console.print("[yellow]Note: ONNX quantized model will run on CPU[/yellow]")
        
        try:
            self.evaluator = ONNXModelEvaluator(onnx_model_path)
            console.print(f"[green]Successfully loaded ONNX quantized model[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load ONNX model: {e}[/red]")
            raise e
    
    def predict(self, data_loader):
        """Get ONNX quantized model predictions"""
        return self.evaluator.predict(data_loader)

def get_onnx_quantized_model():
    """Get the ONNX quantized model path"""
    
    # Use the specific ONNX quantized model path from mobile quantization
    onnx_path = Path("mobile_models/shufflenet_v2_mobile_quantized.onnx")
    
    if onnx_path.exists():
        console.print(f"Found ONNX quantized model at: {onnx_path}")
        return "shufflenet_v2_onnx_quantized", onnx_path
    
    # Fallback: search for ONNX quantized model
    console.print("[yellow]Specific ONNX quantized model not found, searching for ONNX quantized model[/yellow]")
    
    mobile_models_dir = Path("mobile_models")
    if mobile_models_dir.exists():
        mobile_dirs = list(mobile_models_dir.glob("*_quantized.onnx"))
        if mobile_dirs:
            onnx_model = mobile_dirs[0]
            # Extract model name from filename
            model_name = onnx_model.stem.replace("_mobile_quantized", "")
            console.print(f"Found ONNX quantized model: {model_name} at {onnx_model}")
            return f"{model_name}_onnx_quantized", onnx_model
    
    console.print("[red]No ONNX quantized model found[/red]")
    return None, None

def evaluate_onnx_quantized_on_test_dataset(model_preds, model_targets, model_probs, 
                                          class_mapping, dataset_name, save_dir, 
                                          model_name, avg_inference_time, model_info):
    """Evaluate ONNX quantized model performance on a test dataset"""
    
    # Create reverse class mapping
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Calculate metrics
    accuracy = accuracy_score(model_targets, model_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        model_targets, model_preds, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        model_targets, model_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        model_targets, model_preds, average='weighted', zero_division=0
    )
    
    # Create performance dictionary
    performance = {
        "dataset": dataset_name,
        "model": f"{model_name}_onnx_quantized",
        "total_samples": len(model_targets),
        "accuracy": accuracy * 100,
        "precision_macro": precision_macro * 100,
        "recall_macro": recall_macro * 100,
        "f1_macro": f1_macro * 100,
        "precision_weighted": precision_weighted * 100,
        "recall_weighted": recall_weighted * 100,
        "f1_weighted": f1_weighted * 100,
        "avg_inference_time_per_batch": avg_inference_time,
        "inference_device": "CPU",
        "is_quantized": True,
        "model_type": "onnx_quantized",
        "per_class_metrics": {}
    }
    
    # Add model information
    performance.update(model_info)
    
    # Per-class metrics
    unique_targets = np.unique(model_targets)
    for target_idx in unique_targets:
        if target_idx < len(precision):
            class_name = idx_to_class.get(target_idx, f"Class_{target_idx}")
            performance["per_class_metrics"][class_name] = {
                "precision": precision[target_idx] * 100,
                "recall": recall[target_idx] * 100,
                "f1_score": f1[target_idx] * 100,
                "support": int(support[target_idx])
            }
    
    # Print results
    console.print(f"\n[bold]ONNX Quantized Model ({model_name}) Performance on {dataset_name} Test Dataset:[/bold]")
    console.print(f"Total Samples: {len(model_targets)}")
    console.print(f"Accuracy: {accuracy*100:.2f}%")
    console.print(f"Precision (Macro): {precision_macro*100:.2f}%")
    console.print(f"Recall (Macro): {recall_macro*100:.2f}%")
    console.print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
    console.print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
    console.print(f"Average Inference Time per Batch: {avg_inference_time:.4f}s")
    console.print(f"Model Size: {model_info.get('model_size_mb', 0):.2f} MB")
    
    # Print per-class results
    console.print(f"\n[bold]Per-Class Results for {dataset_name}:[/bold]")
    for target_idx in sorted(unique_targets):
        if target_idx < len(precision):
            class_name = idx_to_class.get(target_idx, f"Class_{target_idx}")
            console.print(f"  {class_name}: F1={f1[target_idx]*100:.2f}%, Support={int(support[target_idx])}")
    
    # Save detailed classification report
    try:
        # Get all unique classes in both targets and predictions
        all_unique_classes = sorted(set(model_targets) | set(model_preds))
        class_labels_for_report = [idx_to_class.get(i, f"Class_{i}") for i in all_unique_classes]
        
        report = classification_report(
            model_targets, model_preds, 
            labels=all_unique_classes,
            target_names=class_labels_for_report,
            output_dict=True, 
            zero_division=0
        )
    except Exception as e:
        console.print(f"[yellow]Could not generate classification report with class names: {e}[/yellow]")
        # Fallback: generate report without target names
        report = classification_report(
            model_targets, model_preds, 
            output_dict=True, 
            zero_division=0
        )
    
    with open(save_dir / f"{dataset_name}_onnx_quantized_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Create and save confusion matrix
    cm = confusion_matrix(model_targets, model_preds)
    
    plt.figure(figsize=(12, 10))
    # Use only classes that appear in targets for confusion matrix labels
    target_class_labels = [idx_to_class.get(i, f"Class_{i}") for i in sorted(unique_targets)]
    
    # For confusion matrix, we need to handle potential class mismatch
    try:
        if len(target_class_labels) == cm.shape[0] and len(target_class_labels) == cm.shape[1]:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_class_labels, yticklabels=target_class_labels)
        else:
            # Use numeric labels if there's a mismatch
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            console.print(f"[yellow]Used numeric labels for confusion matrix due to class mismatch[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Confusion matrix visualization error: {e}[/yellow]")
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
    plt.title(f'ONNX Quantized Model ({model_name}) on {dataset_name} Test Dataset - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / f"{dataset_name}_onnx_quantized_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze misclassifications
    misclassified = []
    for i, (pred, target) in enumerate(zip(model_preds, model_targets)):
        if pred != target:
            # Handle cases where predicted class might not be in mapping
            pred_class_name = idx_to_class.get(pred, f"Unknown_Class_{pred}")
            target_class_name = idx_to_class.get(target, f"Unknown_Class_{target}")
            
            misclassified.append({
                'sample_idx': i,
                'predicted_class': pred_class_name,
                'actual_class': target_class_name,
                'confidence': float(np.max(model_probs[i]))
            })
    
    # Save misclassification analysis
    with open(save_dir / f"{dataset_name}_onnx_quantized_misclassifications.json", "w") as f:
        json.dump(misclassified[:100], f, indent=4)  # Save top 100 misclassifications
    
    console.print(f"Misclassified samples: {len(misclassified)}/{len(model_targets)} ({len(misclassified)/len(model_targets)*100:.1f}%)")
    
    # Save performance metrics
    with open(save_dir / f"{dataset_name}_onnx_quantized_performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=4)
    
    return performance

def create_onnx_comparative_analysis(all_results, save_dir):
    """Create comparative analysis between datasets"""
    
    console.print("\n[bold]Creating ONNX Quantized Model Comparative Analysis...[/bold]")
    
    # Prepare data for comparison
    comparison_data = {
        "datasets": list(all_results.keys()),
        "accuracies": [all_results[dataset]["accuracy"] for dataset in all_results],
        "f1_macro": [all_results[dataset]["f1_macro"] for dataset in all_results],
        "f1_weighted": [all_results[dataset]["f1_weighted"] for dataset in all_results],
        "total_samples": [all_results[dataset]["total_samples"] for dataset in all_results],
        "inference_times": [all_results[dataset]["avg_inference_time_per_batch"] for dataset in all_results]
    }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ONNX Quantized Model Performance Comparison Across Test Datasets', fontsize=16)
    
    # Accuracy comparison
    axes[0, 0].bar(comparison_data["datasets"], comparison_data["accuracies"], color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim(0, 100)
    for i, v in enumerate(comparison_data["accuracies"]):
        axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # F1 Score comparison
    x = np.arange(len(comparison_data["datasets"]))
    width = 0.35
    axes[0, 1].bar(x - width/2, comparison_data["f1_macro"], width, label='F1 Macro', color='lightgreen')
    axes[0, 1].bar(x + width/2, comparison_data["f1_weighted"], width, label='F1 Weighted', color='orange')
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(comparison_data["datasets"])
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 100)
    
    # Sample size comparison
    axes[1, 0].bar(comparison_data["datasets"], comparison_data["total_samples"], color=['gold', 'lightsteelblue'])
    axes[1, 0].set_title('Test Sample Size Comparison')
    axes[1, 0].set_ylabel('Number of Samples')
    for i, v in enumerate(comparison_data["total_samples"]):
        axes[1, 0].text(i, v + max(comparison_data["total_samples"])*0.01, f'{v}', ha='center', va='bottom')
    
    # Inference time comparison
    axes[1, 1].bar(comparison_data["datasets"], comparison_data["inference_times"], color=['thistle', 'lightpink'])
    axes[1, 1].set_title('Inference Time Comparison')
    axes[1, 1].set_ylabel('Time per Batch (seconds)')
    for i, v in enumerate(comparison_data["inference_times"]):
        axes[1, 1].text(i, v + max(comparison_data["inference_times"])*0.01, f'{v:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / "onnx_quantized_comparative_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison summary
    with open(save_dir / "onnx_quantized_comparative_summary.json", "w") as f:
        json.dump(comparison_data, f, indent=4)
    
    console.print(f"✓ Comparative analysis saved to {save_dir}")

def main():
    """Evaluate the ONNX quantized model on individual test datasets"""
    
    console.print("[bold]Evaluating ONNX Quantized Model on Individual Test Datasets[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find the ONNX quantized model
    console.print("\n[bold]Step 1: Finding ONNX Quantized Model[/bold]")
    model_name, onnx_model_path = get_onnx_quantized_model()
    
    if model_name is None or onnx_model_path is None:
        console.print("[red]No ONNX quantized model found.[/red]")
        return False
    
    console.print(f"ONNX Quantized Model: {model_name}")
    console.print(f"Model Path: {onnx_model_path}")
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = Path("outputs") / "test_evaluation" / "onnx_quantized_individual" / f"onnx_quantized_individual_test_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Create config
    config = {
        "evaluation_type": "onnx_quantized_individual_test",
        "model": model_name,
        "model_path": str(onnx_model_path),
        "quantized": True,
        "model_format": "onnx",
        "device": "CPU",
        "output_dir": str(run_dir),
        "datasets": ["tomatovillage", "plantvillage"]
    }
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize wandb if available
    run = None
    if WANDB_AVAILABLE and wandb is not None:
        try:
            wandb_name = f"onnx_quantized_individual_test_eval_{timestamp}"
            run = wandb.init(
                project="tomato-disease-test-evaluation",
                name=wandb_name,
                config=config
            )
            console.print("✓ WandB initialized successfully")
        except Exception as e:
            console.print(f"Warning: WandB initialization failed: {e}")
            run = None
    else:
        console.print("WandB not available - logging will be skipped")
    
    all_results = {}
    
    try:
        # Test datasets to evaluate
        test_datasets = ["tomatovillage", "plantvillage"]
        
        # Create ONNX quantized model evaluator
        console.print("\n[bold]Step 2: Creating ONNX Quantized Model Evaluator[/bold]")
        try:
            model_evaluator = ONNXModelTestEvaluator(onnx_model_path)
        except Exception as e:
            console.print(f"[red]Failed to create evaluator: {e}[/red]")
            return False
        
        # Get model information once
        model_info = model_evaluator.evaluator.get_model_info()
        
        for dataset_name in test_datasets:
            console.print(f"\n[bold]Evaluating on {dataset_name.upper()} Test Dataset[/bold]")
            
            # Load test dataset
            console.print(f"\n[bold]Step 3: Loading {dataset_name} Test Dataset[/bold]")
            
            try:
                loader = TestDatasetLoader(dataset_name)
                test_loader, class_mapping = loader.get_test_loader()
                
                console.print(f"✓ {dataset_name} test dataset loaded successfully")
                console.print(f"Test samples: {len(test_loader.dataset)}")
                console.print(f"Classes: {len(class_mapping)}")
                
                # Save class mapping for this dataset
                with open(run_dir / f"{dataset_name}_class_mapping.json", "w") as f:
                    json.dump(class_mapping, f, indent=4)
                
            except Exception as e:
                console.print(f"[red]Failed to load {dataset_name} test dataset: {e}[/red]")
                continue
            
            # Get ONNX quantized model predictions
            console.print(f"\n[bold]Step 4: Getting ONNX Quantized Model Predictions for {dataset_name}[/bold]")
            try:
                model_preds, model_targets, model_probs, avg_inference_time = model_evaluator.predict(test_loader)
            except Exception as e:
                console.print(f"[red]Failed to get predictions for {dataset_name}: {e}[/red]")
                continue
            
            if len(model_preds) == 0:
                console.print(f"[yellow]No predictions generated for {dataset_name}, skipping...[/yellow]")
                continue
            
            # Evaluate ONNX quantized model
            console.print(f"\n[bold]Step 5: Evaluating ONNX Quantized Model Performance on {dataset_name}[/bold]")
            try:
                performance = evaluate_onnx_quantized_on_test_dataset(
                    model_preds, model_targets, model_probs,
                    class_mapping, dataset_name, run_dir, 
                    "shufflenet_v2", avg_inference_time, model_info
                )
                
                all_results[dataset_name] = performance
            except Exception as e:
                console.print(f"[red]Failed to evaluate {dataset_name}: {e}[/red]")
                continue
            
            # Log to wandb
            if WANDB_AVAILABLE and wandb is not None:
                wandb.log({
                    f"{dataset_name}_onnx_quantized_accuracy": performance["accuracy"],
                    f"{dataset_name}_onnx_quantized_f1_macro": performance["f1_macro"],
                    f"{dataset_name}_onnx_quantized_f1_weighted": performance["f1_weighted"],
                    f"{dataset_name}_onnx_quantized_precision_macro": performance["precision_macro"],
                    f"{dataset_name}_onnx_quantized_recall_macro": performance["recall_macro"],
                    f"{dataset_name}_onnx_quantized_total_samples": performance["total_samples"],
                    f"{dataset_name}_onnx_quantized_inference_time": performance["avg_inference_time_per_batch"]
                })
            else:
                print("WandB logging skipped - wandb not available")
        
        # Create comparative analysis
        if len(all_results) > 1:
            console.print("\n[bold]Step 6: Creating Comparative Analysis[/bold]")
            try:
                create_onnx_comparative_analysis(all_results, run_dir)
            except Exception as e:
                console.print(f"Warning: Comparative analysis failed: {e}")
        
        # Save overall evaluation summary
        evaluation_summary = {
            "evaluation_type": "onnx_quantized_individual_test",
            "model": model_name,
            "model_path": str(onnx_model_path),
            "results_by_dataset": all_results,
            "model_info": model_info,
            "evaluation_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4)
        
        # Print final summary
        console.print(f"\n[bold]ONNX Quantized Model Individual Test Evaluation Summary:[/bold]")
        for dataset_name, performance in all_results.items():
            console.print(f"\n{dataset_name.upper()}:")
            console.print(f"  Accuracy: {performance['accuracy']:.2f}%")
            console.print(f"  F1 Score (Macro): {performance['f1_macro']:.2f}%")
            console.print(f"  F1 Score (Weighted): {performance['f1_weighted']:.2f}%")
            console.print(f"  Test Samples: {performance['total_samples']}")
            console.print(f"  Inference Time: {performance['avg_inference_time_per_batch']:.4f}s per batch")
        
        console.print(f"\nModel Size: {model_info.get('model_size_mb', 0):.2f} MB (ONNX Quantized)")
        
        console.print(f"\n[green]ONNX quantized model evaluation on individual test datasets completed successfully![/green]")
        console.print(f"Results saved to: {run_dir}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ ONNX quantized model individual test evaluation failed with exception: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Close wandb run
        if run is not None:
            try:
                wandb.finish()
            except:
                pass

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 