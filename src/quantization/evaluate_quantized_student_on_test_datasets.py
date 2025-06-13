#!/usr/bin/env python3
"""
Evaluate Quantized Student Model on Test Datasets
Tests the quantized student model on test sets of tomatoVillage and plantVillage datasets
Note: Quantized models must run on CPU
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
from test_dataset_loader import (
    create_test_dataloader, 
    analyze_test_dataset, 
    visualize_test_samples,
    get_available_test_datasets
)

console = Console()

class QuantizedStudentTestEvaluator:
    """Quantized student model evaluator for test datasets"""
    
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
        all_samples = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, samples) in enumerate(data_loader):
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
                all_samples.extend(samples)
                
                if batch_idx % 50 == 0:
                    console.print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        # Calculate average inference time per batch
        avg_inference_time = np.mean(inference_times)
        console.print(f"Average inference time per batch: {avg_inference_time:.4f}s")
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs), all_samples, avg_inference_time

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

def evaluate_quantized_student_on_test_dataset(student_preds, student_targets, student_probs, 
                                              all_samples, class_mapping, save_dir, dataset_name, 
                                              student_model_name, avg_inference_time):
    """Evaluate quantized student model performance on test dataset"""
    
    # Create class name mappings
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
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
        "dataset": dataset_name,
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
    console.print(f"\n[bold]Quantized Student Model ({student_model_name}) Performance on {dataset_name} Test Dataset:[/bold]")
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
    
    with open(save_dir / f"{dataset_name}_quantized_student_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Create and save confusion matrix
    cm = confusion_matrix(student_targets, student_preds)
    
    plt.figure(figsize=(12, 10))
    # Use only classes that appear in targets for confusion matrix labels
    target_class_labels = [idx_to_class[i] for i in sorted(unique_targets)]
    pred_classes = sorted(set(student_preds))
    
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
        
    plt.title(f'Quantized Student Model ({student_model_name}) Performance on {dataset_name} Test Dataset - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / f"{dataset_name}_quantized_student_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze misclassifications
    misclassified = []
    for i, (pred, target, sample) in enumerate(zip(student_preds, student_targets, all_samples)):
        if pred != target:
            # Handle cases where predicted class might not be in mapping
            pred_class_name = idx_to_class.get(pred, f"Unknown_Class_{pred}")
            target_class_name = idx_to_class.get(target, f"Unknown_Class_{target}")
            
            misclassified.append({
                'sample_idx': i,
                'predicted_class': pred_class_name,
                'actual_class': target_class_name,
                'original_class_name': sample['original_class_name'],
                'confidence': float(np.max(student_probs[i]))
            })
    
    # Save misclassification analysis
    with open(save_dir / f"{dataset_name}_quantized_student_misclassifications.json", "w") as f:
        json.dump(misclassified[:100], f, indent=4)  # Save top 100 misclassifications
    
    console.print(f"Misclassified samples: {len(misclassified)}/{len(student_targets)} ({len(misclassified)/len(student_targets)*100:.1f}%)")
    
    # Calculate model efficiency metrics (approximation for quantized model)
    base_model = ModelFactory.get_model(student_model_name, len(class_mapping), pretrained=False)
    num_params = sum(p.numel() for p in base_model.parameters())
    
    performance["model_efficiency"] = {
        "total_parameters": int(num_params),
        "parameters_millions": round(num_params / 1e6, 2),
        "quantized": True,
        "runs_on_device": "CPU"
    }
    
    # Save performance metrics
    with open(save_dir / f"{dataset_name}_quantized_student_performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=4)
    
    # Log to wandb
    wandb.log({
        f"{dataset_name}_quantized_student_accuracy": accuracy * 100,
        f"{dataset_name}_quantized_student_f1_macro": f1_macro * 100,
        f"{dataset_name}_quantized_student_f1_weighted": f1_weighted * 100,
        f"{dataset_name}_quantized_student_precision_macro": precision_macro * 100,
        f"{dataset_name}_quantized_student_recall_macro": recall_macro * 100,
        f"{dataset_name}_quantized_student_total_samples": len(student_targets),
        f"{dataset_name}_quantized_student_misclassification_rate": len(misclassified)/len(student_targets)*100,
        f"{dataset_name}_quantized_student_inference_time": avg_inference_time,
        f"quantized_student_model_parameters": num_params
    })
    
    return performance

def evaluate_single_dataset(dataset_name: str, student_evaluator, class_mapping, save_dir, student_model_name):
    """Evaluate quantized student model on a single test dataset"""
    
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
    image_size = get_image_size(student_model_name)  # Use model-specific image size
    test_loader, test_dataset = create_test_dataloader(
        dataset_name, class_mapping, image_size=image_size, batch_size=32
    )
    
    # Get quantized student predictions
    console.print(f"\n[bold]Getting Quantized Student Model Predictions on {dataset_name}[/bold]")
    student_preds, student_targets, student_probs, all_samples, avg_inference_time = student_evaluator.predict(test_loader)
    
    # Evaluate quantized student model
    console.print(f"\n[bold]Evaluating Quantized Student Model Performance on {dataset_name}[/bold]")
    performance = evaluate_quantized_student_on_test_dataset(
        student_preds, student_targets, student_probs, all_samples,
        class_mapping, save_dir, dataset_name, student_model_name, avg_inference_time
    )
    
    return {
        "dataset_stats": dataset_stats,
        "performance": performance
    }

def main():
    """Evaluate the quantized student model on test datasets"""
    
    console.print("[bold]Evaluating Quantized Student Model on Test Datasets[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get available test datasets
    available_datasets = get_available_test_datasets()
    console.print(f"[bold]Available Test Datasets:[/bold] {', '.join(available_datasets)}")
    
    if not available_datasets:
        console.print("[red]No test datasets found. Please ensure test directories exist.[/red]")
        return False
    
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
    run_dir = Path("outputs") / "test_evaluation" / "quantized_student" / f"quantized_student_on_test_datasets_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Load original class mapping
    console.print("\n[bold]Step 2: Loading Original Class Mapping[/bold]")
    class_mapping = load_original_class_mapping()
    console.print(f"Dataset classes: {len(class_mapping)}")
    
    # Create config
    config = {
        "evaluation_type": "quantized_student_on_test_datasets",
        "student_model": student_model_name,
        "model_path": str(quantized_model_path),
        "test_datasets": available_datasets,
        "total_classes": len(class_mapping),
        "quantized": True,
        "device": "CPU",
        "output_dir": str(run_dir)
    }
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize wandb
    wandb_name = f"quantized_student_test_eval_{timestamp}"
    run = wandb.init(
        project="tomato-disease-test-evaluation",
        name=wandb_name,
        config=config
    )
    
    try:
        # Create quantized student model evaluator
        console.print("\n[bold]Step 3: Creating Quantized Student Model Evaluator[/bold]")
        student_evaluator = QuantizedStudentTestEvaluator(
            model_name=student_model_name,
            model_path=quantized_model_path,
            num_classes=len(class_mapping)
        )
        
        # Evaluate on each test dataset
        results = {}
        for dataset_name in available_datasets:
            console.print(f"\n[bold]Step 4.{len(results)+1}: Evaluating on {dataset_name}[/bold]")
            result = evaluate_single_dataset(dataset_name, student_evaluator, class_mapping, run_dir, student_model_name)
            if result is not None:
                results[dataset_name] = result
        
        # Save overall evaluation summary
        evaluation_summary = {
            "evaluation_type": "quantized_student_on_test_datasets",
            "student_model": student_model_name,
            "model_path": str(quantized_model_path),
            "evaluated_datasets": list(results.keys()),
            "results": results,
            "evaluation_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "test_evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4)
        
        # Print summary
        console.print(f"\n[bold]Quantized Student Model Test Evaluation Summary:[/bold]")
        for dataset_name, result in results.items():
            performance = result["performance"]
            console.print(f"  {dataset_name}:")
            console.print(f"    F1 Score (Macro): {performance['f1_macro']:.2f}%")
            console.print(f"    Accuracy: {performance['accuracy']:.2f}%")
            console.print(f"    Test Samples: {performance['total_samples']}")
            console.print(f"    Inference Time per Batch: {performance['avg_inference_time_per_batch']:.4f}s")
        
        if results:
            first_result = list(results.values())[0]
            model_params = first_result["performance"]["model_efficiency"]["parameters_millions"]
            console.print(f"  Model Parameters: {model_params}M (Quantized)")
        
        console.print(f"\n[green]Quantized student model evaluation on test datasets completed successfully![/green]")
        console.print(f"Results saved to: {run_dir}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Quantized student model test evaluation failed with exception: {e}[/red]")
        raise e
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 