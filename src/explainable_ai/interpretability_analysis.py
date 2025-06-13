#!/usr/bin/env python3
"""
Interpretability Analysis for Student Model
Uses GradCAM++ and LIME to provide explainable AI insights
Generates heatmaps and decision boundary visualizations for paper inclusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from rich.console import Console
import warnings
warnings.filterwarnings('ignore')

# XAI libraries
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Installing pytorch-grad-cam...")
    import subprocess
    subprocess.run(["pip", "install", "grad-cam"], check=True)
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

try:
    import lime
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
except ImportError:
    print("Installing LIME...")
    import subprocess
    subprocess.run(["pip", "install", "lime"], check=True)
    import lime
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

from torchvision import transforms
from sklearn.model_selection import train_test_split

# Import existing modules
from config import *
from models import ModelFactory
from test_dataset_loader import (
    create_test_dataloader, 
    get_available_test_datasets,
    TestDatasetFromSplits
)

console = Console()

def debug_model_structure(model, model_name="Unknown"):
    """Debug helper to print model structure"""
    console.print(f"[bold]Model Structure for {model_name}:[/bold]")
    console.print(f"Model type: {type(model).__name__}")
    
    # Print top-level attributes
    console.print("\n[bold]Top-level attributes:[/bold]")
    for name in dir(model):
        if not name.startswith('_') and hasattr(model, name):
            attr = getattr(model, name)
            if isinstance(attr, (nn.Module, nn.Sequential)):
                console.print(f"  {name}: {type(attr).__name__}")
    
    # Print children
    console.print("\n[bold]Model children:[/bold]")
    for i, child in enumerate(model.children()):
        console.print(f"  [{i}]: {type(child).__name__}")
    
    # Print named modules (first few levels)
    console.print("\n[bold]Named modules (first 10):[/bold]")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 10:
            console.print("  ... (truncated)")
            break
        if name:  # Skip the root module
            console.print(f"  {name}: {type(module).__name__}")

class StudentModelInterpreter:
    """Interpretability analyzer for the student model"""
    
    def __init__(self, model_name: str, model_path: Path, class_mapping: Dict[str, int], device: str):
        self.model_name = model_name
        self.model_path = model_path
        self.class_mapping = class_mapping
        self.device = device
        self.idx_to_class = {v: k for k, v in class_mapping.items()}
        
        # Load model
        console.print(f"Loading student model: {model_name}")
        self.model = self._load_model()
        self.model.eval()
        
        # Debug model structure if needed
        debug_model_structure(self.model, model_name)
        
        # Get target layer for GradCAM++
        try:
            self.target_layer = self._get_target_layer()
            console.print(f"[green]Target layer selected: {type(self.target_layer).__name__}[/green]")
        except Exception as e:
            console.print(f"[red]Error selecting target layer: {e}[/red]")
            raise
        
        # Initialize GradCAM++
        try:
            self.gradcam = GradCAMPlusPlus(
                model=self.model,
                target_layers=[self.target_layer],
                use_cuda=device != 'cpu'
            )
            console.print(f"[green]GradCAM++ initialized successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing GradCAM++: {e}[/red]")
            raise
        
        # Initialize LIME
        self.lime_explainer = lime_image.LimeImageExplainer()
        
        console.print(f"[green]Interpretability analyzer initialized successfully[/green]")
    
    def _load_model(self):
        """Load the student model"""
        model = ModelFactory.get_model(self.model_name, len(self.class_mapping), pretrained=False).to(self.device)
        
        # Load state dict
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _get_target_layer(self):
        """Get the appropriate target layer for GradCAM++"""
        
        # First, let's inspect the model structure
        console.print(f"[blue]Inspecting {self.model_name} model structure...[/blue]")
        
        if 'shufflenet' in self.model_name.lower():
            # For ShuffleNet models, try different possible layer names
            possible_layers = [
                'features',  # Common in torchvision models
                'conv_last',  # Sometimes used in custom implementations
                'stage4',     # ShuffleNet stage naming
                'backbone',   # Custom wrapper naming
            ]
            
            for layer_name in possible_layers:
                if hasattr(self.model, layer_name):
                    layer = getattr(self.model, layer_name)
                    console.print(f"[green]Found {layer_name} layer for ShuffleNet[/green]")
                    
                    # If it's a sequential, get the last layer
                    if hasattr(layer, '__len__') and len(layer) > 0:
                        # Get the last convolutional layer in the sequence
                        for i in range(len(layer) - 1, -1, -1):
                            if isinstance(layer[i], (nn.Conv2d, nn.Sequential)):
                                console.print(f"[green]Using layer {layer_name}[{i}] as target[/green]")
                                return layer[i]
                    
                    # If it's already a conv layer, use it
                    elif isinstance(layer, nn.Conv2d):
                        console.print(f"[green]Using {layer_name} as target layer[/green]")
                        return layer
                    
                    # If it's a sequential or module, try to find conv layers inside
                    else:
                        for name, module in layer.named_modules():
                            if isinstance(module, nn.Conv2d):
                                console.print(f"[green]Using {layer_name}.{name} as target layer[/green]")
                                return module
            
            console.print(f"[yellow]Standard ShuffleNet layers not found, using generic approach[/yellow]")
        
        elif 'mobilenet' in self.model_name.lower():
            if hasattr(self.model, 'features'):
                return self.model.features[-1]
        elif 'efficientnet' in self.model_name.lower():
            if hasattr(self.model, 'features'):
                return self.model.features[-1]
        
        # Generic approach - find the last convolutional layer in the entire model
        console.print("[yellow]Using generic target layer detection[/yellow]")
        last_conv_layer = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
                last_conv_name = name
        
        if last_conv_layer is not None:
            console.print(f"[green]Found last conv layer: {last_conv_name}[/green]")
            return last_conv_layer
        
        # Final fallback - use the last module that's not the classifier
        console.print("[red]Could not find conv layer, using fallback[/red]")
        modules = list(self.model.children())
        
        # Try to avoid the classifier (usually the last layer)
        for i in range(len(modules) - 2, -1, -1):
            module = modules[i]
            if isinstance(module, (nn.Conv2d, nn.Sequential)):
                console.print(f"[yellow]Using fallback layer: {type(module).__name__}[/yellow]")
                return module
        
        # Last resort
        console.print("[red]Using very last module as fallback[/red]")
        return modules[-2] if len(modules) > 1 else modules[0]
    
    def predict_single(self, image_tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """Get prediction for a single image"""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred_class].item()
            prob_array = probs.cpu().numpy()[0]
        
        return pred_class, confidence, prob_array
    
    def generate_gradcam_explanation(self, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate GradCAM++ explanation"""
        # Prepare input
        input_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Set target class
        if target_class is None:
            # Use predicted class
            pred_class, _, _ = self.predict_single(image_tensor)
            target_class = pred_class
        
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate GradCAM++
        grayscale_cam = self.gradcam(input_tensor=input_tensor, targets=targets)
        
        return grayscale_cam[0]  # Return first (and only) result
    
    def generate_lime_explanation(self, image_pil: Image.Image, num_features: int = 10, num_samples: int = 1000) -> Tuple[np.ndarray, List]:
        """Generate LIME explanation"""
        # Convert PIL to numpy
        image_np = np.array(image_pil)
        
        # Define prediction function for LIME
        def predict_fn(images):
            """Prediction function for LIME"""
            batch_preds = []
            
            # Transform function
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            for img in images:
                img_tensor = transform(img.astype(np.uint8))
                _, _, probs = self.predict_single(img_tensor)
                batch_preds.append(probs)
            
            return np.array(batch_preds)
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            image_np,
            predict_fn,
            top_labels=len(self.class_mapping),
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features
        )
        
        # Get the explanation for the top predicted class
        pred_class, _, _ = self.predict_single(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image_pil))
        
        temp, mask = explanation.get_image_and_mask(
            pred_class, 
            positive_only=True, 
            num_features=num_features, 
            hide_rest=False
        )
        
        return temp, mask, explanation
    
    def create_combined_visualization(self, 
                                    original_image: Image.Image,
                                    gradcam_heatmap: np.ndarray,
                                    lime_explanation: np.ndarray,
                                    lime_mask: np.ndarray,
                                    pred_class: int,
                                    confidence: float,
                                    actual_class: str,
                                    save_path: Path) -> None:
        """Create a comprehensive visualization combining all explanations"""
        
        # Prepare high-quality original image
        original_size = (320, 320)  # Higher resolution for better clarity
        original_resized = original_image.resize(original_size, Image.Resampling.LANCZOS)
        original_np = np.array(original_resized) / 255.0
        
        # Resize heatmap to match original image size for better overlay
        gradcam_resized = cv2.resize(gradcam_heatmap, original_size)
        
        # Create high-quality GradCAM++ overlay with better transparency
        gradcam_overlay = show_cam_on_image(original_np, gradcam_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)
        
        # Create enhanced LIME overlay with clear boundaries
        lime_resized = cv2.resize(lime_explanation, original_size)
        lime_mask_resized = cv2.resize(lime_mask.astype(np.float32), original_size)
        
        # Create LIME boundary overlay
        lime_boundary_overlay = original_np.copy()
        
        # Create contours for LIME regions
        # Normalize lime_mask_resized to 0-1 range
        mask_normalized = (lime_mask_resized - lime_mask_resized.min()) / (lime_mask_resized.max() - lime_mask_resized.min() + 1e-8)
        
        # Create binary mask for important regions (positive contributions)
        important_regions = mask_normalized > 0.1
        
        # Find contours
        important_uint8 = (important_regions * 255).astype(np.uint8)
        contours, _ = cv2.findContours(important_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw boundaries on the image
        lime_boundary_overlay = (lime_boundary_overlay * 255).astype(np.uint8)
        cv2.drawContours(lime_boundary_overlay, contours, -1, (0, 255, 0), 3)  # Green boundaries
        lime_boundary_overlay = lime_boundary_overlay.astype(np.float32) / 255.0
        
        # Create LIME importance overlay with transparent color mapping
        lime_importance_overlay = original_np.copy()
        
        # Apply color mapping to important regions
        # Red for positive importance, blue for negative
        positive_mask = mask_normalized > 0.2
        negative_mask = mask_normalized < -0.2
        
        # Create color overlay
        color_overlay = np.zeros_like(original_np)
        color_overlay[positive_mask] = [1.0, 0.0, 0.0]  # Red for important
        color_overlay[negative_mask] = [0.0, 0.0, 1.0]  # Blue for less important
        
        # Blend with original image
        alpha = 0.4
        lime_importance_overlay = (1 - alpha) * original_np + alpha * color_overlay
        
        # Create figure with better layout and larger size
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('white')
        
        # Original image - high quality
        axes[0, 0].imshow(original_resized)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # GradCAM++ attention overlay
        axes[0, 1].imshow(gradcam_overlay)
        axes[0, 1].set_title('GradCAM++ Attention Map', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Enhanced GradCAM++ heatmap only (for reference)
        im1 = axes[0, 2].imshow(gradcam_resized, cmap='jet', alpha=0.8)
        axes[0, 2].set_title('GradCAM++ Heatmap', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        # Add colorbar for heatmap
        cbar1 = plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        cbar1.set_label('Attention Intensity', fontsize=10)
        
        # LIME decision boundaries
        axes[1, 0].imshow(lime_boundary_overlay)
        axes[1, 0].set_title('LIME Decision Boundaries', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # LIME feature importance overlay
        axes[1, 1].imshow(lime_importance_overlay)
        axes[1, 1].set_title('LIME Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Prediction results and analysis
        axes[1, 2].axis('off')
        pred_class_name = self.idx_to_class[pred_class]
        
        # Create text with better formatting
        correct_pred = "✓" if pred_class_name.replace("Tomato___", "") == actual_class.replace("Tomato___", "") else "✗"
        
        info_text = f"""PREDICTION ANALYSIS
        
🔍 Model: {self.model_name.upper()}

📊 RESULTS:
  Actual: {actual_class}
  Predicted: {pred_class_name}
  Confidence: {confidence:.1f}%
  Status: {correct_pred}

🧠 INTERPRETABILITY:
  • GradCAM++: Shows model attention
  • LIME: Explains decision regions
  
🎯 FOCUS AREAS:
  • Red regions: Important for decision
  • Green boundaries: LIME segments
  • Hot colors: High attention
"""
        
        axes[1, 2].text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top', 
                        transform=axes[1, 2].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Overall title with better formatting
        status_color = "green" if correct_pred == "✓" else "red"
        fig.suptitle(f'Interpretability Analysis: {actual_class} (Confidence: {confidence:.1f}%)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.2)
        
        # Save with high quality
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        console.print(f"[green]Saved enhanced visualization: {save_path}[/green]")

def find_best_kd_model():
    """Find the trained knowledge distillation student model"""
    
    # Look for the specific KD model path
    kd_path = Path("outputs/knowledge_distillation/shufflenet_v2/best_kd_distillation_20250530152128")
    model_file = kd_path / "best_model.pth"
    
    if model_file.exists():
        console.print(f"Found KD model at: {model_file}")
        return "shufflenet_v2", model_file
    
    # Fallback: search for most recent KD model
    kd_base_dir = Path("outputs") / "knowledge_distillation"
    if kd_base_dir.exists():
        for student_dir in kd_base_dir.iterdir():
            if student_dir.is_dir():
                kd_runs = [d for d in student_dir.iterdir() 
                          if d.is_dir() and "best_kd" in d.name]
                
                if kd_runs:
                    latest_run = max(kd_runs, key=lambda x: x.stat().st_mtime)
                    model_file = latest_run / "best_model.pth"
                    
                    if model_file.exists():
                        student_model_name = student_dir.name
                        return student_model_name, model_file
    
    return None, None

def load_original_class_mapping():
    """Load the original combined dataset class mapping"""
    
    class_mapping_files = list(Path("outputs").rglob("class_mapping.json"))
    
    for mapping_file in class_mapping_files:
        if "combined" in str(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
    
    if class_mapping_files:
        with open(class_mapping_files[0], 'r') as f:
            return json.load(f)
    
    # Fallback to default mapping
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

def select_representative_samples(dataset_name: str, class_mapping: Dict[str, int], samples_per_class: int = 3) -> Dict[str, List]:
    """Select representative samples from each class in the dataset"""
    
    console.print(f"Selecting representative samples from {dataset_name}")
    
    # Create test dataset
    dataset = TestDatasetFromSplits(dataset_name, class_mapping)
    
    if len(dataset.test_samples) == 0:
        console.print(f"[yellow]No test samples found for {dataset_name}[/yellow]")
        return {}
    
    # Group samples by original class
    class_samples = {}
    for sample in dataset.test_samples:
        original_class = sample['original_class_name']
        if original_class not in class_samples:
            class_samples[original_class] = []
        class_samples[original_class].append(sample)
    
    # Select representative samples
    selected_samples = {}
    for class_name, samples in class_samples.items():
        # Sort by confidence or randomly select
        np.random.seed(42)  # For reproducibility
        selected = np.random.choice(samples, min(samples_per_class, len(samples)), replace=False)
        selected_samples[class_name] = selected.tolist()
    
    total_selected = sum(len(samples) for samples in selected_samples.values())
    console.print(f"Selected {total_selected} representative samples from {len(selected_samples)} classes")
    
    return selected_samples

def analyze_dataset_interpretability(dataset_name: str, interpreter: StudentModelInterpreter, 
                                   selected_samples: Dict[str, List], save_dir: Path) -> Dict:
    """Analyze interpretability for a specific dataset"""
    
    console.print(f"\n[bold]Analyzing interpretability for {dataset_name}[/bold]")
    
    # Create dataset-specific directory
    dataset_dir = save_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    analysis_results = {
        'dataset': dataset_name,
        'total_samples_analyzed': 0,
        'classes_analyzed': [],
        'interpretability_metrics': {}
    }
    
    for class_name, samples in selected_samples.items():
        console.print(f"  Analyzing class: {class_name} ({len(samples)} samples)")
        
        class_dir = dataset_dir / class_name.replace('/', '_').replace(' ', '_')
        class_dir.mkdir(exist_ok=True)
        
        class_results = {
            'samples_analyzed': len(samples),
            'avg_confidence': 0,
            'correct_predictions': 0
        }
        
        for i, sample in enumerate(samples):
            try:
                # Load and preprocess image
                original_image = Image.open(sample['image_path']).convert('RGB')
                image_tensor = transform(original_image)
                
                # Get prediction
                pred_class, confidence, _ = interpreter.predict_single(image_tensor)
                pred_class_name = interpreter.idx_to_class[pred_class]
                
                # Check if prediction is correct
                is_correct = pred_class == sample['combined_label']
                if is_correct:
                    class_results['correct_predictions'] += 1
                
                class_results['avg_confidence'] += confidence
                
                # Generate GradCAM++ explanation
                console.print(f"    Generating GradCAM++ for sample {i+1}/{len(samples)}")
                gradcam_heatmap = interpreter.generate_gradcam_explanation(image_tensor, pred_class)
                
                # Generate LIME explanation
                console.print(f"    Generating LIME for sample {i+1}/{len(samples)}")
                lime_explanation, lime_mask, lime_obj = interpreter.generate_lime_explanation(
                    original_image, num_features=10, num_samples=1000
                )
                
                # Create combined visualization
                save_path = class_dir / f"interpretability_sample_{i+1}.png"
                interpreter.create_combined_visualization(
                    original_image=original_image,
                    gradcam_heatmap=gradcam_heatmap,
                    lime_explanation=lime_explanation,
                    lime_mask=lime_mask,
                    pred_class=pred_class,
                    confidence=confidence * 100,
                    actual_class=class_name,
                    save_path=save_path
                )
                
                # Save individual explanations for detailed analysis
                individual_dir = class_dir / f"sample_{i+1}_details"
                individual_dir.mkdir(exist_ok=True)
                
                # Save high-quality original image
                original_image.resize((320, 320), Image.Resampling.LANCZOS).save(individual_dir / "original.png")
                
                # Create and save enhanced GradCAM++ overlay
                original_np_hq = np.array(original_image.resize((320, 320), Image.Resampling.LANCZOS)) / 255.0
                gradcam_hq = cv2.resize(gradcam_heatmap, (320, 320))
                gradcam_overlay_hq = show_cam_on_image(original_np_hq, gradcam_hq, use_rgb=True, colormap=cv2.COLORMAP_JET)
                
                # Save GradCAM++ overlay
                plt.figure(figsize=(8, 8))
                plt.imshow(gradcam_overlay_hq)
                plt.axis('off')
                plt.title(f'GradCAM++ Attention - {class_name}', fontsize=14, fontweight='bold', pad=20)
                plt.savefig(individual_dir / "gradcam_overlay.png", dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                # Save standalone GradCAM++ heatmap with colorbar
                plt.figure(figsize=(8, 8))
                im = plt.imshow(gradcam_hq, cmap='jet')
                plt.axis('off')
                plt.title(f'GradCAM++ Heatmap - {class_name}', fontsize=14, fontweight='bold', pad=20)
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Intensity', fontsize=12)
                plt.savefig(individual_dir / "gradcam_heatmap.png", dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                
                # Create and save enhanced LIME visualizations
                lime_resized_hq = cv2.resize(lime_explanation, (320, 320))
                lime_mask_hq = cv2.resize(lime_mask.astype(np.float32), (320, 320))
                
                # LIME decision boundaries
                lime_boundary = original_np_hq.copy()
                mask_norm = (lime_mask_hq - lime_mask_hq.min()) / (lime_mask_hq.max() - lime_mask_hq.min() + 1e-8)
                important_regions = mask_norm > 0.1
                important_uint8 = (important_regions * 255).astype(np.uint8)
                contours, _ = cv2.findContours(important_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                lime_boundary_vis = (lime_boundary * 255).astype(np.uint8)
                cv2.drawContours(lime_boundary_vis, contours, -1, (0, 255, 0), 4)  # Thicker green boundaries
                lime_boundary_vis = lime_boundary_vis.astype(np.float32) / 255.0
                
                plt.figure(figsize=(8, 8))
                plt.imshow(lime_boundary_vis)
                plt.axis('off')
                plt.title(f'LIME Decision Boundaries - {class_name}', fontsize=14, fontweight='bold', pad=20)
                plt.savefig(individual_dir / "lime_boundaries.png", dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                
                # LIME importance overlay
                lime_importance = original_np_hq.copy()
                positive_mask = mask_norm > 0.2
                negative_mask = mask_norm < -0.2
                
                color_overlay = np.zeros_like(original_np_hq)
                color_overlay[positive_mask] = [1.0, 0.0, 0.0]  # Red for important
                color_overlay[negative_mask] = [0.0, 0.0, 1.0]  # Blue for less important
                
                alpha = 0.4
                lime_importance_vis = (1 - alpha) * original_np_hq + alpha * color_overlay
                
                plt.figure(figsize=(8, 8))
                plt.imshow(lime_importance_vis)
                plt.axis('off')
                plt.title(f'LIME Feature Importance - {class_name}', fontsize=14, fontweight='bold', pad=20)
                plt.savefig(individual_dir / "lime_importance.png", dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                
                console.print(f"    [green]Completed sample {i+1}/{len(samples)}[/green]")
                
            except Exception as e:
                console.print(f"    [red]Error processing sample {i+1}: {e}[/red]")
                continue
        
        # Calculate class-level metrics
        if len(samples) > 0:
            class_results['avg_confidence'] /= len(samples)
            class_results['accuracy'] = class_results['correct_predictions'] / len(samples)
        
        analysis_results['classes_analyzed'].append(class_name)
        analysis_results['interpretability_metrics'][class_name] = class_results
        analysis_results['total_samples_analyzed'] += len(samples)
    
    # Save analysis results
    with open(dataset_dir / "interpretability_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=4)
    
    return analysis_results

def create_summary_visualization(all_results: Dict, save_dir: Path):
    """Create summary visualizations across all datasets"""
    
    console.print("\n[bold]Creating summary visualizations[/bold]")
    
    # Extract metrics for visualization
    datasets = []
    classes = []
    accuracies = []
    confidences = []
    
    for dataset_name, results in all_results.items():
        for class_name, metrics in results['interpretability_metrics'].items():
            datasets.append(dataset_name)
            classes.append(class_name)
            accuracies.append(metrics['accuracy'] * 100)
            confidences.append(metrics['avg_confidence'] * 100)
    
    # Create accuracy comparison plot
    plt.figure(figsize=(15, 8))
    
    # Group by dataset
    dataset_names = list(all_results.keys())
    for i, dataset_name in enumerate(dataset_names):
        dataset_accuracies = [acc for j, acc in enumerate(accuracies) if datasets[j] == dataset_name]
        dataset_classes = [cls for j, cls in enumerate(classes) if datasets[j] == dataset_name]
        
        x_positions = np.arange(len(dataset_classes)) + i * 0.35
        plt.bar(x_positions, dataset_accuracies, width=0.35, label=dataset_name, alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Student Model Accuracy by Class and Dataset')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confidence distribution plot
    plt.figure(figsize=(12, 6))
    
    for dataset_name in dataset_names:
        dataset_confidences = [conf for j, conf in enumerate(confidences) if datasets[j] == dataset_name]
        plt.hist(dataset_confidences, alpha=0.7, label=dataset_name, bins=20)
    
    plt.xlabel('Confidence (%)')
    plt.ylabel('Frequency')
    plt.title('Student Model Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Summary visualizations saved to: {save_dir}[/green]")

def main():
    """Main interpretability analysis pipeline"""
    
    console.print("[bold]Student Model Interpretability Analysis[/bold]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find the trained KD model
    console.print("\n[bold]Step 1: Finding Trained Student Model[/bold]")
    student_model_name, kd_model_path = find_best_kd_model()
    
    if student_model_name is None or kd_model_path is None:
        console.print("[red]No trained student model found. Please run knowledge distillation first[/red]")
        return False
    
    console.print(f"Student Model: {student_model_name}")
    console.print(f"Model Path: {kd_model_path}")
    
    # Load class mapping
    console.print("\n[bold]Step 2: Loading Class Mapping[/bold]")
    class_mapping = load_original_class_mapping()
    console.print(f"Total classes: {len(class_mapping)}")
    
    # Create interpretability analyzer
    console.print("\n[bold]Step 3: Initializing Interpretability Analyzer[/bold]")
    interpreter = StudentModelInterpreter(
        model_name=student_model_name,
        model_path=kd_model_path,
        class_mapping=class_mapping,
        device=DEVICE
    )
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_dir = Path("outputs") / "interpretability" / f"student_interpretability_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Results will be saved to:[/bold] {save_dir}")
    
    # Get available test datasets
    available_datasets = get_available_test_datasets()
    console.print(f"[bold]Available datasets:[/bold] {', '.join(available_datasets)}")
    
    if not available_datasets:
        console.print("[red]No test datasets found[/red]")
        return False
    
    # Analyze each dataset
    all_results = {}
    for dataset_name in available_datasets:
        console.print(f"\n[bold]Step 4: Analyzing {dataset_name}[/bold]")
        
        # Select representative samples
        selected_samples = select_representative_samples(dataset_name, class_mapping, samples_per_class=3)
        
        if not selected_samples:
            console.print(f"[yellow]No samples found for {dataset_name}, skipping[/yellow]")
            continue
        
        # Analyze interpretability
        results = analyze_dataset_interpretability(
            dataset_name, interpreter, selected_samples, save_dir
        )
        all_results[dataset_name] = results
    
    # Create summary visualizations
    if all_results:
        console.print("\n[bold]Step 5: Creating Summary Visualizations[/bold]")
        create_summary_visualization(all_results, save_dir)
        
        # Save overall summary
        overall_summary = {
            "analysis_type": "student_model_interpretability",
            "student_model": student_model_name,
            "model_path": str(kd_model_path),
            "total_datasets_analyzed": len(all_results),
            "total_samples_analyzed": sum(r['total_samples_analyzed'] for r in all_results.values()),
            "datasets_results": all_results,
            "analysis_completed_at": datetime.now().isoformat()
        }
        
        with open(save_dir / "interpretability_summary.json", "w") as f:
            json.dump(overall_summary, f, indent=4)
        
        # Print summary
        console.print(f"\n[bold]Interpretability Analysis Summary:[/bold]")
        console.print(f"  Student Model: {student_model_name}")
        console.print(f"  Datasets Analyzed: {len(all_results)}")
        console.print(f"  Total Samples: {overall_summary['total_samples_analyzed']}")
        
        for dataset_name, results in all_results.items():
            console.print(f"  {dataset_name}: {results['total_samples_analyzed']} samples, {len(results['classes_analyzed'])} classes")
        
        console.print(f"\n[green]Interpretability analysis completed successfully![/green]")
        console.print(f"Results saved to: {save_dir}")
        console.print(f"\n[bold]Generated visualizations include:[/bold]")
        console.print("  - Combined GradCAM++ and LIME explanations for each sample")
        console.print("  - Individual heatmaps and explanation images")
        console.print("  - Accuracy and confidence distribution plots")
        console.print("  - Detailed analysis JSON files")
        
        return True
    
    else:
        console.print("[red]No datasets could be analyzed[/red]")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 