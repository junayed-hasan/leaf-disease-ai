#!/usr/bin/env python3
"""
Evaluate ONNX Quantized Student Model on Combined Test Dataset
Tests the ONNX quantized student model on the combined test dataset using the same
data split and preprocessing as used during training
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
    # Define basic fallbacks
    def get_dataset_config(name):
        return {"num_classes": 15}
    def get_image_size(model):
        return 224

try:
    from onnx_model_evaluator import ONNXModelEvaluator, ONNXModelAnalyzer
except ImportError as e:
    print(f"Error: Could not import ONNX evaluator: {e}")
    print("Make sure onnx_model_evaluator.py is in the same directory")
    sys.exit(1)

try:
    from dataset_balancing import load_balanced_data, analyze_balanced_dataset
except ImportError:
    print("Warning: dataset_balancing not available")
    load_balanced_data = None
    analyze_balanced_dataset = None

try:
    from dataset import visualize_samples, load_data
except ImportError as e:
    print(f"Warning: Could not import dataset functions: {e}")
    def visualize_samples(*args, **kwargs):
        print("Visualization skipped - dataset module not available")
    def load_data(dataset_name, model_name, augmentation_config=None):
        print(f"Error: load_data function not available")
        raise ImportError("dataset module not available")

try:
    from torch.utils.data import DataLoader
except ImportError:
    print("Error: PyTorch DataLoader not available")
    sys.exit(1)

class ONNXStudentCombinedTestEvaluator:
    """ONNX quantized student model evaluator for combined test dataset"""
    
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

def get_onnx_quantized_student_model():
    """Get the ONNX quantized student model path"""
    
    # Use the specific ONNX quantized model path from mobile quantization
    onnx_path = Path("mobile_models/shufflenet_v2_mobile_quantized.onnx")
    
    if onnx_path.exists():
        console.print(f"Found ONNX quantized student model at: {onnx_path}")
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
    
    console.print("[red]No ONNX quantized student model found[/red]")
    return None, None

def evaluate_onnx_quantized_on_combined_test(student_preds, student_targets, student_probs, 
                                           class_to_idx, save_dir, student_model_name, avg_inference_time, model_info):
    """Evaluate ONNX quantized student model performance on combined test dataset"""
    
    # Create class name mappings
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Calculate metrics
    accuracy = accuracy_score(student_targets, student_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        student_targets, student_preds, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        student_targets, student_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        student_targets, student_preds, average='weighted', zero_division=0
    )
    
    # Create performance dictionary
    performance = {
        "dataset": "combined",
        "student_model": f"{student_model_name}_onnx_quantized",
        "total_samples": len(student_targets),
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
    unique_targets = np.unique(student_targets)
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
    console.print(f"\n[bold]ONNX Quantized Student Model ({student_model_name}) Performance on Combined Test Dataset:[/bold]")
    console.print(f"Total Samples: {len(student_targets)}")
    console.print(f"Accuracy: {accuracy*100:.2f}%")
    console.print(f"Precision (Macro): {precision_macro*100:.2f}%")
    console.print(f"Recall (Macro): {recall_macro*100:.2f}%")
    console.print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
    console.print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
    console.print(f"Average Inference Time per Batch: {avg_inference_time:.4f}s")
    console.print(f"Model Size: {model_info.get('model_size_mb', 0):.2f} MB")
    
    # Print per-class results
    console.print(f"\n[bold]Per-Class Results:[/bold]")
    for target_idx in sorted(unique_targets):
        if target_idx < len(precision):
            class_name = idx_to_class[target_idx]
            console.print(f"  {class_name}: F1={f1[target_idx]*100:.2f}%, Support={int(support[target_idx])}")
    
    # Save detailed classification report
    try:
        # Get all unique classes in both targets and predictions
        all_unique_classes = sorted(set(student_targets) | set(student_preds))
        class_labels_for_report = [idx_to_class.get(i, f"Class_{i}") for i in all_unique_classes]
        
        report = classification_report(
            student_targets, student_preds, 
            labels=all_unique_classes,
            target_names=class_labels_for_report,
            output_dict=True, 
            zero_division=0
        )
    except Exception as e:
        console.print(f"[yellow]Could not generate classification report with class names: {e}[/yellow]")
        # Fallback: generate report without target names
        report = classification_report(
            student_targets, student_preds, 
            output_dict=True, 
            zero_division=0
        )
    
    with open(save_dir / "combined_onnx_quantized_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Create and save confusion matrix
    cm = confusion_matrix(student_targets, student_preds)
    
    plt.figure(figsize=(15, 12))
    # Use only classes that appear in targets for confusion matrix labels
    target_class_labels = [idx_to_class[i] for i in sorted(unique_targets)]
    
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
        
    plt.title(f'ONNX Quantized Student Model ({student_model_name}) Performance on Combined Test Dataset - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "combined_onnx_quantized_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze misclassifications
    misclassified = []
    for i, (pred, target) in enumerate(zip(student_preds, student_targets)):
        if pred != target:
            # Handle cases where predicted class might not be in mapping
            pred_class_name = idx_to_class.get(pred, f"Unknown_Class_{pred}")
            target_class_name = idx_to_class.get(target, f"Unknown_Class_{target}")
            
            misclassified.append({
                'sample_idx': i,
                'predicted_class': pred_class_name,
                'actual_class': target_class_name,
                'confidence': float(np.max(student_probs[i]))
            })
    
    # Save misclassification analysis
    with open(save_dir / "combined_onnx_quantized_misclassifications.json", "w") as f:
        json.dump(misclassified[:100], f, indent=4)  # Save top 100 misclassifications
    
    console.print(f"Misclassified samples: {len(misclassified)}/{len(student_targets)} ({len(misclassified)/len(student_targets)*100:.1f}%)")
    
    # Save performance metrics
    with open(save_dir / "combined_onnx_quantized_performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=4)
    
    # Log to wandb
    if WANDB_AVAILABLE and wandb is not None:
        wandb.log({
            "combined_onnx_quantized_accuracy": accuracy * 100,
            "combined_onnx_quantized_f1_macro": f1_macro * 100,
            "combined_onnx_quantized_f1_weighted": f1_weighted * 100,
            "combined_onnx_quantized_precision_macro": precision_macro * 100,
            "combined_onnx_quantized_recall_macro": recall_macro * 100,
            "combined_onnx_quantized_total_samples": len(student_targets),
            "combined_onnx_quantized_misclassification_rate": len(misclassified)/len(student_targets)*100,
            "combined_onnx_quantized_inference_time": avg_inference_time,
            "onnx_quantized_model_size_mb": model_info.get('model_size_mb', 0)
        })
    else:
        print("WandB logging skipped - wandb not available")
    
    return performance

def main():
    """Evaluate the ONNX quantized student model on combined test dataset"""
    
    console.print("[bold]Evaluating ONNX Quantized Student Model on Combined Test Dataset[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dataset configuration
    dataset_name = "combined"
    
    # Find the ONNX quantized student model
    console.print("\n[bold]Step 1: Finding ONNX Quantized Student Model[/bold]")
    student_model_name, onnx_model_path = get_onnx_quantized_student_model()
    
    if student_model_name is None or onnx_model_path is None:
        console.print("[red]No ONNX quantized student model found.[/red]")
        return False
    
    console.print(f"ONNX Quantized Student Model: {student_model_name}")
    console.print(f"Model Path: {onnx_model_path}")
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = Path("outputs") / "test_evaluation" / "onnx_quantized_combined" / f"onnx_quantized_combined_test_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Get dataset configuration
    try:
        dataset_config = get_dataset_config(dataset_name)
        num_classes = dataset_config['num_classes']
    except:
        num_classes = 15  # fallback
    
    # Get model-specific image size
    try:
        image_size = get_image_size("shufflenet_v2")  # Use base model name for image size
    except:
        image_size = 224  # fallback
    
    # Create config
    config = {
        "evaluation_type": "onnx_quantized_combined_test",
        "student_model": student_model_name,
        "model_path": str(onnx_model_path),
        "dataset": dataset_name,
        "num_classes": num_classes,
        "image_size": image_size,
        "quantized": True,
        "model_format": "onnx",
        "device": "CPU",
        "output_dir": str(run_dir)
    }
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize wandb if available
    run = None
    if WANDB_AVAILABLE and wandb is not None:
        try:
            wandb_name = f"onnx_quantized_combined_test_eval_{timestamp}"
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
    
    try:
        # Analyze dataset
        console.print("\n[bold]Step 2: Analyzing Combined Dataset[/bold]")
        console.print(f"Using original {dataset_name} dataset without resampling")
        
        # Simple dataset info instead of full analysis
        dataset_info = {
            "dataset_name": dataset_name,
            "num_classes": num_classes,
            "image_size": image_size,
            "balancing_applied": False,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save dataset info
        with open(run_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=4)
        
        # Load combined dataset with same balancing as training
        console.print("\n[bold]Step 3: Loading Combined Test Data[/bold]")
        console.print("Loading original test data without resampling for evaluation")
        
        # Load data using the correct function signature
        try:
            train_loader, val_loader, test_loader, class_to_idx = load_data(
                dataset_name, 
                "shufflenet_v2",  # Use base model name for proper transforms
                augmentation_config=None  # No augmentation for testing
            )
        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            return False
        
        # test_loader is already a DataLoader, no need to create another one
        console.print(f"Test dataset loaded: {len(test_loader.dataset)} samples")
        console.print(f"Number of classes: {len(class_to_idx)}")
        
        # Save class mapping
        with open(run_dir / "class_mapping.json", "w") as f:
            json.dump(class_to_idx, f, indent=4)
        
        # Visualize samples
        console.print("\n[bold]Step 4: Visualizing Test Samples[/bold]")
        try:
            visualize_samples(dataset_name, class_to_idx, run_dir)
        except Exception as e:
            console.print(f"Warning: Sample visualization failed: {e}")
        
        # Create ONNX quantized student model evaluator
        console.print("\n[bold]Step 5: Creating ONNX Quantized Student Model Evaluator[/bold]")
        try:
            student_evaluator = ONNXStudentCombinedTestEvaluator(onnx_model_path)
        except Exception as e:
            console.print(f"[red]Failed to create evaluator: {e}[/red]")
            return False
        
        # Get ONNX quantized student predictions
        console.print("\n[bold]Step 6: Getting ONNX Quantized Student Model Predictions[/bold]")
        try:
            student_preds, student_targets, student_probs, avg_inference_time = student_evaluator.predict(test_loader)
        except Exception as e:
            console.print(f"[red]Failed to get predictions: {e}[/red]")
            return False
        
        # Get model information
        model_info = student_evaluator.evaluator.get_model_info()
        
        # Evaluate ONNX quantized student model
        console.print("\n[bold]Step 7: Evaluating ONNX Quantized Student Model Performance[/bold]")
        performance = evaluate_onnx_quantized_on_combined_test(
            student_preds, student_targets, student_probs,
            class_to_idx, run_dir, "shufflenet_v2", avg_inference_time, model_info
        )
        
        # Save overall evaluation summary
        evaluation_summary = {
            "evaluation_type": "onnx_quantized_combined_test",
            "student_model": student_model_name,
            "model_path": str(onnx_model_path),
            "dataset": dataset_name,
            "performance": performance,
            "dataset_statistics": dataset_info,
            "model_info": model_info,
            "evaluation_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4)
        
        # Print summary
        console.print(f"\n[bold]ONNX Quantized Student Model Combined Test Evaluation Summary:[/bold]")
        console.print(f"Dataset: {dataset_name}")
        console.print(f"F1 Score (Macro): {performance['f1_macro']:.2f}%")
        console.print(f"F1 Score (Weighted): {performance['f1_weighted']:.2f}%")
        console.print(f"Accuracy: {performance['accuracy']:.2f}%")
        console.print(f"Test Samples: {performance['total_samples']}")
        console.print(f"Inference Time per Batch: {performance['avg_inference_time_per_batch']:.4f}s")
        console.print(f"Model Size: {model_info.get('model_size_mb', 0):.2f} MB (ONNX Quantized)")
        
        console.print(f"\n[green]ONNX quantized student model evaluation on combined test dataset completed successfully![/green]")
        console.print(f"Results saved to: {run_dir}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ ONNX quantized student model combined test evaluation failed with exception: {e}[/red]")
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
        sys.exit(1) 