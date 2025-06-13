#!/usr/bin/env python3
"""
Evaluate Quantized Student Model on Combined Test Dataset
Tests the quantized student model on the combined test dataset using the same
data split and preprocessing as used during training
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
import time

# Import existing modules
from config import *
from models import ModelFactory
from dataset_balancing import load_balanced_data, analyze_balanced_dataset
from dataset import visualize_samples
from dataset import load_data
from torch.utils.data import DataLoader

console = Console()

class QuantizedStudentCombinedTestEvaluator:
    """Quantized student model evaluator for combined test dataset"""
    
    def __init__(self, model_name, model_path, num_classes):
        self.model_name = model_name
        self.model_path = model_path
        self.num_classes = num_classes
        # Quantized models must run on CPU
        self.device = torch.device('cpu')
        
        # Load the quantized student model
        console.print(f"Loading quantized student model {model_name} from {model_path}")
        console.print("[yellow]Note: Quantized model will run on CPU for compatibility[/yellow]")
        self.model = self._load_quantized_model()
        self.model.eval()  # Set to evaluation mode
        
        console.print(f"[green]Successfully loaded quantized student model: {model_name}[/green]")
    
    def _load_quantized_model(self):
        """Load the quantized student model"""
        try:
            # Load quantized model directly
            model_data = torch.load(self.model_path, map_location='cpu')
            
            # Check if it's a state dict or a model
            if hasattr(model_data, 'eval'):
                # It's already a model
                console.print("Loaded quantized model directly")
                return model_data
            else:
                # It's a state dict, need to clean and load it
                console.print("Loading quantized state dict into model")
                
                # Clean the state dict by removing FLOPs-related keys
                cleaned_state_dict = {}
                for key, value in model_data.items():
                    # Skip FLOPs calculation keys
                    if 'total_ops' not in key and 'total_params' not in key:
                        # Skip quantization-specific keys that don't belong to the model
                        if not any(skip_key in key for skip_key in ['scale', 'zero_point', '_packed_params', 'dtype']):
                            cleaned_state_dict[key] = value
                
                console.print(f"Cleaned state dict: removed {len(model_data) - len(cleaned_state_dict)} FLOPs/quantization metadata keys")
                
                # Try to load as a regular model first and then quantize
                base_model = ModelFactory.get_model(self.model_name, self.num_classes, pretrained=False)
                
                # Load the cleaned state dict
                try:
                    base_model.load_state_dict(cleaned_state_dict, strict=False)
                    console.print("Loaded cleaned state dict into base model")
                except Exception as e:
                    console.print(f"[yellow]Could not load cleaned state dict: {e}[/yellow]")
                    # Try alternative approach - load from the original KD model
                    kd_model_path = Path("outputs/knowledge_distillation/shufflenet_v2/best_kd_distillation_20250530152128/best_model.pth")
                    if kd_model_path.exists():
                        console.print("Loading from original KD model and applying quantization")
                        kd_checkpoint = torch.load(kd_model_path, map_location='cpu')
                        if 'model_state_dict' in kd_checkpoint:
                            base_model.load_state_dict(kd_checkpoint['model_state_dict'])
                        else:
                            base_model.load_state_dict(kd_checkpoint)
                    else:
                        raise Exception("Could not load model weights")
                
                # Apply dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    base_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                console.print("Applied dynamic quantization to model")
                
                return quantized_model
                
        except Exception as e:
            console.print(f"[red]Error loading quantized model: {e}[/red]")
            console.print("[yellow]Attempting to load from original KD model and quantize...[/yellow]")
            
            # Fallback: load from the original KD model and apply quantization
            try:
                kd_model_path = Path("outputs/knowledge_distillation/shufflenet_v2/best_kd_distillation_20250530152128/best_model.pth")
                if not kd_model_path.exists():
                    raise Exception("Original KD model not found")
                
                base_model = ModelFactory.get_model(self.model_name, self.num_classes, pretrained=False)
                kd_checkpoint = torch.load(kd_model_path, map_location='cpu')
                
                if 'model_state_dict' in kd_checkpoint:
                    base_model.load_state_dict(kd_checkpoint['model_state_dict'])
                else:
                    base_model.load_state_dict(kd_checkpoint)
                
                console.print("Loaded original KD model")
                
                # Apply dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    base_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                console.print("Applied fresh quantization to KD model")
                
                return quantized_model
                
            except Exception as fallback_e:
                console.print(f"[red]Fallback loading also failed: {fallback_e}[/red]")
                raise Exception(f"Could not load quantized model: {e}, Fallback failed: {fallback_e}")
    
    def predict(self, data_loader):
        """Get quantized student model predictions"""
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # Move data to CPU for quantized model
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                # Get predictions from quantized student model
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    console.print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        # Calculate average inference time per batch
        avg_inference_time = np.mean(inference_times)
        console.print(f"Average inference time per batch: {avg_inference_time:.4f}s")
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs), avg_inference_time

def get_quantized_student_model():
    """Get the quantized student model path"""
    
    # Use the specific quantized model path provided by user
    quantized_path = Path("outputs/computational_analysis/analysis_20250601061137/shufflenet_v2_quantized.pth")
    
    if quantized_path.exists():
        console.print(f"Found quantized student model at: {quantized_path}")
        return "shufflenet_v2", quantized_path
    
    # Fallback: search for quantized student model
    console.print("[yellow]Specific quantized model not found, searching for quantized student model[/yellow]")
    
    analysis_dirs = list(Path("outputs").glob("**/analysis_*"))
    for analysis_dir in analysis_dirs:
        quantized_models = list(analysis_dir.glob("*_quantized.pth"))
        if quantized_models:
            quantized_model = quantized_models[0]
            # Extract model name from filename
            model_name = quantized_model.stem.replace("_quantized", "")
            console.print(f"Found quantized model: {model_name} at {quantized_model}")
            return model_name, quantized_model
    
    console.print("[red]No quantized student model found[/red]")
    return None, None

def evaluate_quantized_student_on_combined_test(student_preds, student_targets, student_probs, 
                                               class_to_idx, save_dir, student_model_name, avg_inference_time):
    """Evaluate quantized student model performance on combined test dataset"""
    
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
        "student_model": f"{student_model_name}_quantized",
        "total_samples": len(student_targets),
        "accuracy": accuracy * 100,
        "precision_macro": precision_macro * 100,
        "recall_macro": recall_macro * 100,
        "f1_macro": f1_macro * 100,
        "precision_weighted": precision_weighted * 100,
        "recall_weighted": recall_weighted * 100,
        "f1_weighted": f1_weighted * 100,
        "avg_inference_time_per_batch": avg_inference_time,
        "per_class_metrics": {}
    }
    
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
    console.print(f"\n[bold]Quantized Student Model ({student_model_name}) Performance on Combined Test Dataset:[/bold]")
    console.print(f"Total Samples: {len(student_targets)}")
    console.print(f"Accuracy: {accuracy*100:.2f}%")
    console.print(f"Precision (Macro): {precision_macro*100:.2f}%")
    console.print(f"Recall (Macro): {recall_macro*100:.2f}%")
    console.print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
    console.print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
    console.print(f"Average Inference Time per Batch: {avg_inference_time:.4f}s")
    
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
    
    with open(save_dir / "combined_quantized_student_classification_report.json", "w") as f:
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
        
    plt.title(f'Quantized Student Model ({student_model_name}) Performance on Combined Test Dataset - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "combined_quantized_student_confusion_matrix.png", dpi=300, bbox_inches='tight')
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
    with open(save_dir / "combined_quantized_student_misclassifications.json", "w") as f:
        json.dump(misclassified[:100], f, indent=4)  # Save top 100 misclassifications
    
    console.print(f"Misclassified samples: {len(misclassified)}/{len(student_targets)} ({len(misclassified)/len(student_targets)*100:.1f}%)")
    
    # Calculate model efficiency metrics (approximation for quantized model)
    base_model = ModelFactory.get_model(student_model_name, len(class_to_idx), pretrained=False)
    num_params = sum(p.numel() for p in base_model.parameters())
    
    performance["model_efficiency"] = {
        "total_parameters": int(num_params),
        "parameters_millions": round(num_params / 1e6, 2),
        "quantized": True,
        "runs_on_device": "CPU"
    }
    
    # Save performance metrics
    with open(save_dir / "combined_quantized_student_performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=4)
    
    # Log to wandb
    wandb.log({
        "combined_quantized_student_accuracy": accuracy * 100,
        "combined_quantized_student_f1_macro": f1_macro * 100,
        "combined_quantized_student_f1_weighted": f1_weighted * 100,
        "combined_quantized_student_precision_macro": precision_macro * 100,
        "combined_quantized_student_recall_macro": recall_macro * 100,
        "combined_quantized_student_total_samples": len(student_targets),
        "combined_quantized_student_misclassification_rate": len(misclassified)/len(student_targets)*100,
        "combined_quantized_student_inference_time": avg_inference_time,
        "quantized_student_model_parameters": num_params
    })
    
    return performance

def main():
    """Evaluate the quantized student model on combined test dataset"""
    
    console.print("[bold]Evaluating Quantized Student Model on Combined Test Dataset[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dataset configuration
    dataset_name = "combined"
    
    # Find the quantized student model
    console.print("\n[bold]Step 1: Finding Quantized Student Model[/bold]")
    student_model_name, quantized_model_path = get_quantized_student_model()
    
    if student_model_name is None or quantized_model_path is None:
        console.print("[red]No quantized student model found.[/red]")
        return False
    
    console.print(f"Quantized Student Model: {student_model_name}")
    console.print(f"Model Path: {quantized_model_path}")
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = Path("outputs") / "test_evaluation" / "quantized_student_combined" / f"quantized_student_combined_test_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    num_classes = dataset_config['num_classes']
    
    # Get model-specific image size
    image_size = get_image_size(student_model_name)
    
    # Create config
    config = {
        "evaluation_type": "quantized_student_combined_test",
        "student_model": student_model_name,
        "model_path": str(quantized_model_path),
        "dataset": dataset_name,
        "num_classes": num_classes,
        "image_size": image_size,
        "quantized": True,
        "device": "CPU",
        "output_dir": str(run_dir)
    }
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize wandb
    wandb_name = f"quantized_student_combined_test_eval_{timestamp}"
    run = wandb.init(
        project="tomato-disease-test-evaluation",
        name=wandb_name,
        config=config
    )
    
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
        train_loader, val_loader, test_loader, class_to_idx = load_data(
            dataset_name, 
            student_model_name,  # Use student model name for proper transforms
            augmentation_config=None  # No augmentation for testing
        )
        
        # test_loader is already a DataLoader, no need to create another one
        console.print(f"Test dataset loaded: {len(test_loader.dataset)} samples")
        console.print(f"Number of classes: {len(class_to_idx)}")
        
        # Save class mapping
        with open(run_dir / "class_mapping.json", "w") as f:
            json.dump(class_to_idx, f, indent=4)
        
        # Visualize samples
        console.print("\n[bold]Step 4: Visualizing Test Samples[/bold]")
        visualize_samples(dataset_name, class_to_idx, run_dir)
        
        # Create quantized student model evaluator
        console.print("\n[bold]Step 5: Creating Quantized Student Model Evaluator[/bold]")
        student_evaluator = QuantizedStudentCombinedTestEvaluator(
            model_name=student_model_name,
            model_path=quantized_model_path,
            num_classes=num_classes
        )
        
        # Get quantized student predictions
        console.print("\n[bold]Step 6: Getting Quantized Student Model Predictions[/bold]")
        student_preds, student_targets, student_probs, avg_inference_time = student_evaluator.predict(test_loader)
        
        # Evaluate quantized student model
        console.print("\n[bold]Step 7: Evaluating Quantized Student Model Performance[/bold]")
        performance = evaluate_quantized_student_on_combined_test(
            student_preds, student_targets, student_probs,
            class_to_idx, run_dir, student_model_name, avg_inference_time
        )
        
        # Save overall evaluation summary
        evaluation_summary = {
            "evaluation_type": "quantized_student_combined_test",
            "student_model": student_model_name,
            "model_path": str(quantized_model_path),
            "dataset": dataset_name,
            "performance": performance,
            "dataset_statistics": dataset_info,
            "evaluation_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4)
        
        # Print summary
        console.print(f"\n[bold]Quantized Student Model Combined Test Evaluation Summary:[/bold]")
        console.print(f"Dataset: {dataset_name}")
        console.print(f"F1 Score (Macro): {performance['f1_macro']:.2f}%")
        console.print(f"F1 Score (Weighted): {performance['f1_weighted']:.2f}%")
        console.print(f"Accuracy: {performance['accuracy']:.2f}%")
        console.print(f"Test Samples: {performance['total_samples']}")
        console.print(f"Inference Time per Batch: {performance['avg_inference_time_per_batch']:.4f}s")
        console.print(f"Model Parameters: {performance['model_efficiency']['parameters_millions']}M (Quantized)")
        
        console.print(f"\n[green]Quantized student model evaluation on combined test dataset completed successfully![/green]")
        console.print(f"Results saved to: {run_dir}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Quantized student model combined test evaluation failed with exception: {e}[/red]")
        raise e
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 