#!/usr/bin/env python3
"""
Evaluate Knowledge Distillation Student Model on PlantDoc Dataset
Tests the trained KD student model on external PlantDoc dataset with class mapping
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
from plantdoc_dataset import (
    create_plantdoc_dataloader, 
    analyze_plantdoc_dataset, 
    visualize_plantdoc_samples,
    get_plantdoc_subset_classes,
    PLANTDOC_TO_COMBINED_MAPPING
)

console = Console()

class StudentModelEvaluator:
    """Student model evaluator for PlantDoc dataset"""
    
    def __init__(self, model_name, model_path, num_classes, device):
        self.model_name = model_name
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device
        
        # Load the student model
        console.print(f"Loading student model {model_name} from {model_path}")
        self.model = self._load_model()
        self.model.eval()  # Set to evaluation mode
        
        console.print(f"[green]Successfully loaded student model: {model_name}[/green]")
    
    def _load_model(self):
        """Load the student model"""
        model = ModelFactory.get_model(self.model_name, self.num_classes, pretrained=False).to(self.device)
        
        # Load state dict
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def predict(self, data_loader):
        """Get student model predictions"""
        all_preds = []
        all_targets = []
        all_probs = []
        all_samples = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, samples) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get predictions from student model
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_samples.extend(samples)
                
                if batch_idx % 50 == 0:
                    console.print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs), all_samples

def find_best_kd_model():
    """Find the trained knowledge distillation student model"""
    
    # Look for the specific KD model path provided by user
    kd_path = Path("outputs/knowledge_distillation/shufflenet_v2/best_kd_distillation_20250530152128")
    model_file = kd_path / "best_model.pth"
    
    if model_file.exists():
        console.print(f"Found KD model at: {model_file}")
        return "shufflenet_v2", model_file
    
    # Fallback: search for most recent KD model
    console.print("[yellow]Specific KD model not found, searching for most recent KD model[/yellow]")
    
    kd_base_dir = Path("outputs") / "knowledge_distillation"
    if kd_base_dir.exists():
        for student_dir in kd_base_dir.iterdir():
            if student_dir.is_dir():
                # Find the most recent KD run
                kd_runs = [d for d in student_dir.iterdir() 
                          if d.is_dir() and "best_kd" in d.name]
                
                if kd_runs:
                    latest_run = max(kd_runs, key=lambda x: x.stat().st_mtime)
                    model_file = latest_run / "best_model.pth"
                    
                    if model_file.exists():
                        # Extract student model name from directory
                        student_model_name = student_dir.name
                        console.print(f"Found KD model: {student_model_name} at {model_file}")
                        return student_model_name, model_file
    
    console.print("[red]No KD model found[/red]")
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

def evaluate_student_on_plantdoc(student_preds, student_targets, student_probs, 
                                all_samples, combined_class_mapping, plantdoc_subset_mapping, 
                                save_dir, student_model_name):
    """Evaluate student model performance on PlantDoc dataset"""
    
    # Create class name mappings
    combined_idx_to_class = {v: k for k, v in combined_class_mapping.items()}
    
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
        "dataset": "PlantDoc",
        "student_model": student_model_name,
        "total_samples": len(student_targets),
        "accuracy": accuracy * 100,
        "precision_macro": precision_macro * 100,
        "recall_macro": recall_macro * 100,
        "f1_macro": f1_macro * 100,
        "precision_weighted": precision_weighted * 100,
        "recall_weighted": recall_weighted * 100,
        "f1_weighted": f1_weighted * 100,
        "per_class_metrics": {}
    }
    
    # Per-class metrics (only for classes present in PlantDoc)
    unique_targets = np.unique(student_targets)
    for target_idx in unique_targets:
        if target_idx < len(precision):
            class_name = combined_idx_to_class[target_idx]
            performance["per_class_metrics"][class_name] = {
                "precision": precision[target_idx] * 100,
                "recall": recall[target_idx] * 100,
                "f1_score": f1[target_idx] * 100,
                "support": int(support[target_idx])
            }
    
    # Print results
    console.print(f"\n[bold]Student Model ({student_model_name}) Performance on PlantDoc Dataset:[/bold]")
    console.print(f"Total Samples: {len(student_targets)}")
    console.print(f"Accuracy: {accuracy*100:.2f}%")
    console.print(f"Precision (Macro): {precision_macro*100:.2f}%")
    console.print(f"Recall (Macro): {recall_macro*100:.2f}%")
    console.print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
    console.print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
    
    # Print per-class results
    console.print(f"\n[bold]Per-Class Results:[/bold]")
    for target_idx in sorted(unique_targets):
        if target_idx < len(precision):
            class_name = combined_idx_to_class[target_idx]
            console.print(f"  {class_name}: F1={f1[target_idx]*100:.2f}%, Support={int(support[target_idx])}")
    
    # Save detailed classification report
    report = classification_report(
        student_targets, student_preds, 
        target_names=[combined_idx_to_class[i] for i in sorted(unique_targets)],
        output_dict=True, 
        zero_division=0
    )
    
    with open(save_dir / "plantdoc_student_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Create and save confusion matrix
    cm = confusion_matrix(student_targets, student_preds)
    
    plt.figure(figsize=(12, 10))
    class_labels = [combined_idx_to_class[i] for i in sorted(unique_targets)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Student Model ({student_model_name}) Performance on PlantDoc Dataset - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "plantdoc_student_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze misclassifications
    misclassified = []
    for i, (pred, target, sample) in enumerate(zip(student_preds, student_targets, all_samples)):
        if pred != target:
            misclassified.append({
                'sample_idx': i,
                'predicted_class': combined_idx_to_class[pred],
                'actual_class': combined_idx_to_class[target],
                'plantdoc_class': sample['plantdoc_class'],
                'confidence': float(np.max(student_probs[i]))
            })
    
    # Save misclassification analysis
    with open(save_dir / "plantdoc_student_misclassifications.json", "w") as f:
        json.dump(misclassified[:100], f, indent=4)  # Save top 100 misclassifications
    
    console.print(f"Misclassified samples: {len(misclassified)}/{len(student_targets)} ({len(misclassified)/len(student_targets)*100:.1f}%)")
    
    # Calculate model efficiency metrics
    num_params = sum(p.numel() for p in ModelFactory.get_model(student_model_name, len(combined_class_mapping), pretrained=False).parameters())
    performance["model_efficiency"] = {
        "total_parameters": int(num_params),
        "parameters_millions": round(num_params / 1e6, 2)
    }
    
    # Save performance metrics
    with open(save_dir / "plantdoc_student_performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=4)
    
    # Log to wandb
    wandb.log({
        "plantdoc_student_accuracy": accuracy * 100,
        "plantdoc_student_f1_macro": f1_macro * 100,
        "plantdoc_student_f1_weighted": f1_weighted * 100,
        "plantdoc_student_precision_macro": precision_macro * 100,
        "plantdoc_student_recall_macro": recall_macro * 100,
        "plantdoc_student_total_samples": len(student_targets),
        "plantdoc_student_misclassification_rate": len(misclassified)/len(student_targets)*100,
        "student_model_parameters": num_params
    })
    
    return performance

def main():
    """Evaluate the knowledge distillation student model on PlantDoc dataset"""
    
    console.print("[bold]Evaluating Knowledge Distillation Student Model on PlantDoc Dataset[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    plantdoc_root = "plantdoc"
    
    console.print(f"[bold]PlantDoc Dataset:[/bold] {plantdoc_root}")
    
    # Find the trained KD model
    console.print("\n[bold]Step 1: Finding Trained Knowledge Distillation Model[/bold]")
    student_model_name, kd_model_path = find_best_kd_model()
    
    if student_model_name is None or kd_model_path is None:
        console.print("[red]No trained KD model found. Please run train_best_kd.py first[/red]")
        return False
    
    console.print(f"Student Model: {student_model_name}")
    console.print(f"Model Path: {kd_model_path}")
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = Path("outputs") / "plantdoc_evaluation" / "student" / f"student_on_plantdoc_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Outputs will be saved to:[/bold] {run_dir}")
    
    # Load original class mapping
    console.print("\n[bold]Step 2: Loading Original Class Mapping[/bold]")
    combined_class_mapping = load_original_class_mapping()
    console.print(f"Original dataset classes: {len(combined_class_mapping)}")
    
    # Get PlantDoc subset classes
    plantdoc_subset_mapping = get_plantdoc_subset_classes(combined_class_mapping)
    console.print(f"PlantDoc subset classes: {len(plantdoc_subset_mapping)}")
    console.print("Class mapping:")
    for plantdoc_class, combined_class in PLANTDOC_TO_COMBINED_MAPPING.items():
        combined_idx = combined_class_mapping.get(combined_class, "NOT_FOUND")
        console.print(f"  {plantdoc_class} -> {combined_class} (idx: {combined_idx})")
    
    # Create config
    config = {
        "evaluation_type": "student_on_plantdoc",
        "student_model": student_model_name,
        "model_path": str(kd_model_path),
        "plantdoc_root": plantdoc_root,
        "plantdoc_classes": len(PLANTDOC_TO_COMBINED_MAPPING),
        "original_classes": len(combined_class_mapping),
        "class_mapping": PLANTDOC_TO_COMBINED_MAPPING,
        "output_dir": str(run_dir)
    }
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize wandb
    wandb_name = f"student_plantdoc_eval_{timestamp}"
    run = wandb.init(
        project="tomato-disease-plantdoc-evaluation",
        name=wandb_name,
        config=config
    )
    
    try:
        # Analyze PlantDoc dataset
        console.print("\n[bold]Step 3: Analyzing PlantDoc Dataset[/bold]")
        dataset_stats = analyze_plantdoc_dataset(plantdoc_root, run_dir)
        
        # Visualize PlantDoc samples
        console.print("\n[bold]Step 4: Visualizing PlantDoc Samples[/bold]")
        visualize_plantdoc_samples(plantdoc_root, run_dir)
        
        # Create PlantDoc dataloader
        console.print("\n[bold]Step 5: Loading PlantDoc Data[/bold]")
        image_size = get_image_size(student_model_name)  # Use model-specific image size
        plantdoc_loader, plantdoc_dataset = create_plantdoc_dataloader(
            plantdoc_root, combined_class_mapping, image_size=image_size, batch_size=32
        )
        
        # Create student model evaluator
        console.print("\n[bold]Step 6: Creating Student Model Evaluator[/bold]")
        student_evaluator = StudentModelEvaluator(
            model_name=student_model_name,
            model_path=kd_model_path,
            num_classes=len(combined_class_mapping),
            device=DEVICE
        )
        
        # Get student predictions
        console.print("\n[bold]Step 7: Getting Student Model Predictions[/bold]")
        student_preds, student_targets, student_probs, all_samples = student_evaluator.predict(plantdoc_loader)
        
        # Evaluate student model
        console.print("\n[bold]Step 8: Evaluating Student Model Performance[/bold]")
        performance = evaluate_student_on_plantdoc(
            student_preds, student_targets, student_probs, all_samples,
            combined_class_mapping, plantdoc_subset_mapping, run_dir, student_model_name
        )
        
        # Save evaluation summary
        evaluation_summary = {
            "evaluation_type": "student_on_plantdoc",
            "student_model": student_model_name,
            "model_path": str(kd_model_path),
            "dataset_stats": dataset_stats,
            "performance": performance,
            "evaluation_completed_at": datetime.now().isoformat()
        }
        
        with open(run_dir / "evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4)
        
        console.print(f"\n[green]Student model evaluation on PlantDoc completed successfully![/green]")
        console.print(f"F1 Score (Macro): {performance['f1_macro']:.2f}%")
        console.print(f"Accuracy: {performance['accuracy']:.2f}%")
        console.print(f"Model Parameters: {performance['model_efficiency']['parameters_millions']}M")
        console.print(f"Results saved to: {run_dir}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Student model evaluation failed with exception: {e}[/red]")
        raise e
    finally:
        # Close wandb run
        wandb.finish()
    
    console.print(f"\n[bold]Student Model Evaluation Summary:[/bold]")
    console.print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 