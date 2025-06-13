#!/usr/bin/env python3
"""
ONNX Model Evaluator for Quantized Models
Provides evaluation capabilities for ONNX quantized models
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Global flag to check if ONNX Runtime is available
ONNX_RUNTIME_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    print("Warning: onnxruntime not available. Please install with: pip install onnxruntime")
    ort = None

# Check if ONNX is available for model info
ONNX_AVAILABLE = False
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: onnx not available. Model info will be limited.")
    onnx = None

class ONNXModelEvaluator:
    """Evaluator for ONNX quantized models"""
    
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        self.device = 'cpu'  # ONNX quantized models run on CPU
        
        # Check if ONNX Runtime is available
        if not ONNX_RUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime is required but not installed. "
                "Please install with: pip install onnxruntime"
            )
        
        # Load ONNX model
        self._load_onnx_model()
    
    def _load_onnx_model(self):
        """Load ONNX model with runtime"""
        try:
            # Verify model file exists
            if not Path(self.onnx_model_path).exists():
                raise FileNotFoundError(f"ONNX model not found at: {self.onnx_model_path}")
            
            # Create inference session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set providers with fallback
            providers = ['CPUExecutionProvider']
            
            self.ort_session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output info
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            
            # Get input shape
            input_shape = self.ort_session.get_inputs()[0].shape
            output_shape = self.ort_session.get_outputs()[0].shape
            
            print(f"✓ Loaded ONNX model: {self.onnx_model_path}")
            print(f"Input: {self.input_name} {input_shape}")
            print(f"Output: {self.output_name} {output_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Get ONNX model predictions"""
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []
        
        print(f"Running ONNX inference on {self.device}")
        
        for batch_idx, batch_data in enumerate(data_loader):
            # Handle different batch formats
            if len(batch_data) == 2:
                inputs, targets = batch_data
                samples = None
            else:
                inputs, targets, samples = batch_data
            
            # Convert PyTorch tensor to numpy
            inputs_np = inputs.cpu().numpy().astype(np.float32)
            targets_np = targets.cpu().numpy()
            
            # Measure inference time
            start_time = time.time()
            
            try:
                # Run ONNX inference
                outputs = self.ort_session.run(
                    [self.output_name], 
                    {self.input_name: inputs_np}
                )[0]
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                # Convert outputs to PyTorch for compatibility
                outputs_torch = torch.from_numpy(outputs)
                probs = F.softmax(outputs_torch, dim=1)
                preds = torch.argmax(outputs_torch, dim=1)
                
                all_preds.extend(preds.numpy())
                all_targets.extend(targets_np)
                all_probs.extend(probs.numpy())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
            
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        # Calculate average inference time per batch
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        print(f"Average ONNX inference time per batch: {avg_inference_time:.4f}s")
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs), avg_inference_time
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ONNX model information"""
        try:
            # Basic model info that's always available
            model_size_mb = Path(self.onnx_model_path).stat().st_size / (1024 * 1024)
            
            model_info = {
                'model_path': str(self.onnx_model_path),
                'model_size_mb': round(model_size_mb, 2),
                'input_name': self.input_name,
                'output_name': self.output_name,
                'quantized': True,
                'runtime': 'onnxruntime'
            }
            
            # Add ONNX-specific info if available
            if ONNX_AVAILABLE:
                try:
                    model = onnx.load(self.onnx_model_path)
                    model_info['opset_version'] = model.opset_import[0].version if model.opset_import else 'unknown'
                except:
                    model_info['opset_version'] = 'unknown'
            else:
                model_info['opset_version'] = 'onnx_not_available'
            
            return model_info
            
        except Exception as e:
            return {
                'model_path': str(self.onnx_model_path),
                'model_size_mb': Path(self.onnx_model_path).stat().st_size / (1024 * 1024),
                'error': str(e)
            }

class ONNXModelAnalyzer:
    """Analyzer for ONNX model performance metrics"""
    
    def __init__(self, onnx_evaluator: ONNXModelEvaluator):
        self.evaluator = onnx_evaluator
    
    def analyze_performance(self, data_loader, class_mapping: Dict, save_dir: Path) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        
        print("🔍 Analyzing ONNX model performance...")
        
        # Get predictions
        preds, targets, probs, avg_inference_time = self.evaluator.predict(data_loader)
        
        if len(preds) == 0:
            return {'error': 'No predictions generated'}
        
        # Calculate metrics
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, preds, average='weighted', zero_division=0
        )
        
        # Create performance dictionary
        performance = {
            "model_type": "onnx_quantized",
            "total_samples": len(targets),
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
            "per_class_metrics": {}
        }
        
        # Per-class metrics
        idx_to_class = {v: k for k, v in class_mapping.items()}
        unique_targets = np.unique(targets)
        
        for target_idx in unique_targets:
            if target_idx < len(precision):
                class_name = idx_to_class.get(target_idx, f"Class_{target_idx}")
                performance["per_class_metrics"][class_name] = {
                    "precision": precision[target_idx] * 100,
                    "recall": recall[target_idx] * 100,
                    "f1_score": f1[target_idx] * 100,
                    "support": int(support[target_idx])
                }
        
        # Add model info
        model_info = self.evaluator.get_model_info()
        performance.update(model_info)
        
        # Save detailed classification report
        try:
            all_unique_classes = sorted(set(targets) | set(preds))
            class_labels_for_report = [idx_to_class.get(i, f"Class_{i}") for i in all_unique_classes]
            
            report = classification_report(
                targets, preds, 
                labels=all_unique_classes,
                target_names=class_labels_for_report,
                output_dict=True, 
                zero_division=0
            )
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            report = classification_report(targets, preds, output_dict=True, zero_division=0)
        
        # Save results
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "onnx_classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        with open(save_dir / "onnx_performance_metrics.json", "w") as f:
            json.dump(performance, f, indent=4)
        
        # Analyze misclassifications
        misclassified = []
        for i, (pred, target) in enumerate(zip(preds, targets)):
            if pred != target:
                pred_class_name = idx_to_class.get(pred, f"Unknown_Class_{pred}")
                target_class_name = idx_to_class.get(target, f"Unknown_Class_{target}")
                
                misclassified.append({
                    'sample_idx': i,
                    'predicted_class': pred_class_name,
                    'actual_class': target_class_name,
                    'confidence': float(np.max(probs[i])) if i < len(probs) else 0.0
                })
        
        # Save misclassification analysis
        with open(save_dir / "onnx_misclassifications.json", "w") as f:
            json.dump(misclassified[:100], f, indent=4)  # Save top 100
        
        print(f"✓ ONNX model analysis completed")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"F1 Score (Macro): {f1_macro*100:.2f}%")
        print(f"F1 Score (Weighted): {f1_weighted*100:.2f}%")
        print(f"Inference Time: {avg_inference_time:.4f}s per batch")
        print(f"Misclassified: {len(misclassified)}/{len(targets)} ({len(misclassified)/len(targets)*100:.1f}%)")
        
        return performance 