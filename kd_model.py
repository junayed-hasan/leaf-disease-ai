"""
Knowledge Distillation Model Wrapper
Combines teacher ensemble with student model for distillation training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from ensemble import EnsembleModel
from models import ModelFactory
from config import DEVICE


class KnowledgeDistillationModel(nn.Module):
    """
    Knowledge Distillation wrapper that combines teacher ensemble and student model
    """
    
    def __init__(self, 
                 teacher_models: List[str],
                 teacher_paths: List[str], 
                 student_model_name: str,
                 student_path: Optional[str],
                 num_classes: int,
                 device: str = DEVICE,
                 alpha: float = 1/3,
                 beta: float = 1/3, 
                 gamma: float = 1/3,
                 temperature: float = 4.0,
                 teacher_temp: Optional[float] = None,
                 student_temp: Optional[float] = None,
                 use_feature_distillation: bool = False):
        """
        Initialize KD model
        
        Args:
            teacher_models: List of teacher model names
            teacher_paths: List of paths to teacher model weights
            student_model_name: Name of student model architecture
            student_path: Path to pretrained student weights (optional)
            num_classes: Number of output classes
            device: Device to run on
            alpha: Weight for classification loss
            beta: Weight for logit distillation loss
            gamma: Weight for feature distillation loss
            temperature: Temperature for basic KD (used when temp scaling disabled)
            teacher_temp: Teacher temperature for advanced temp scaling
            student_temp: Student temperature for advanced temp scaling  
            use_feature_distillation: Whether to use feature-level distillation
        """
        super().__init__()
        
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.teacher_temp = teacher_temp if teacher_temp is not None else temperature
        self.student_temp = student_temp if student_temp is not None else 1.0
        self.use_feature_distillation = use_feature_distillation
        self.use_temp_scaling = teacher_temp is not None
        
        # Initialize teacher ensemble
        self.teacher_models = []
        for i, model_name in enumerate(teacher_models):
            # Load checkpoint first to detect number of classes
            checkpoint = torch.load(teacher_paths[i], map_location=device)
            
            # Detect number of classes from checkpoint
            # Look for the final layer weights to determine original num_classes
            teacher_num_classes = None
            for key, tensor in checkpoint['model_state_dict'].items():
                if key.endswith('fc.weight') or key.endswith('classifier.weight') or key.endswith('head.weight'):
                    teacher_num_classes = tensor.shape[0]
                    break
            
            if teacher_num_classes is None:
                # Fallback: try common final layer names
                for key, tensor in checkpoint['model_state_dict'].items():
                    if 'fc' in key and 'weight' in key and len(tensor.shape) == 2:
                        teacher_num_classes = tensor.shape[0]
                        break
                    elif 'classifier' in key and 'weight' in key and len(tensor.shape) == 2:
                        teacher_num_classes = tensor.shape[0]
                        break
            
            if teacher_num_classes is None:
                raise ValueError(f"Could not detect number of classes for teacher model {model_name}")
            
            # Initialize model with correct number of classes
            model = ModelFactory.get_model(model_name, teacher_num_classes, pretrained=True).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.teacher_models.append(model)
            
            print(f"Loaded teacher {model_name} with {teacher_num_classes} classes")
        
        # Freeze teacher models
        for model in self.teacher_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Initialize student model
        self.student_model = ModelFactory.get_model(
            student_model_name, num_classes, pretrained=True
        ).to(device)
        
        # Load pretrained student weights if provided
        if student_path and Path(student_path).exists():
            checkpoint = torch.load(student_path, map_location=device)
            self.student_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained student weights from {student_path}")
        
        # Feature adaptation layer for matching dimensions
        self.teacher_adaptations = None  # Will be initialized in _create_feature_adaptation
        if self.use_feature_distillation:
            self.feature_adaptation = self._create_feature_adaptation()
        
        # Store model info
        self.student_model_name = student_model_name
        self.teacher_model_names = teacher_models
        
    def _create_feature_adaptation(self) -> nn.Module:
        """Create adaptation layers to match student features to teacher features"""
        # First, determine a common feature dimension by checking all teacher models
        teacher_dims = []
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            for model in self.teacher_models:
                features = self._extract_features_from_model(model, dummy_input)
                teacher_dims.append(features.shape[1])
            
            # Get student feature dimension
            student_features = self._extract_student_features(dummy_input)
            student_dim = student_features.shape[1]
        
        # Use the maximum dimension as the common dimension
        common_dim = max(teacher_dims)
        
        # Create adaptation layers for teachers to common dimension
        self.teacher_adaptations = nn.ModuleList()
        for dim in teacher_dims:
            if dim != common_dim:
                self.teacher_adaptations.append(nn.Linear(dim, common_dim).to(self.device))
            else:
                self.teacher_adaptations.append(nn.Identity().to(self.device))
        
        # Create adaptation layer for student to common dimension
        if student_dim != common_dim:
            return nn.Linear(student_dim, common_dim).to(self.device)
        else:
            return nn.Identity().to(self.device)
    
    def _extract_teacher_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from teacher ensemble (penultimate layer)"""
        if self.teacher_adaptations is None:
            # Feature distillation is disabled, create temporary adaptation
            teacher_features = []
            for model in self.teacher_models:
                features = self._extract_features_from_model(model, x)
                teacher_features.append(features)
            
            # If dimensions don't match, use the first teacher's features
            if len(set(f.shape[1] for f in teacher_features)) > 1:
                # Different dimensions, just use first teacher's features
                return teacher_features[0]
            else:
                # Same dimensions, can stack and average
                ensemble_features = torch.stack(teacher_features, dim=0).mean(dim=0)
                return ensemble_features
        else:
            # Feature distillation is enabled, use adapted features
            adapted_teacher_features = []
            
            for i, model in enumerate(self.teacher_models):
                # Extract features from penultimate layer
                features = self._extract_features_from_model(model, x)
                # Apply teacher-specific adaptation
                adapted_features = self.teacher_adaptations[i](features)
                adapted_teacher_features.append(adapted_features)
            
            # Average ensemble features (now all have the same dimension)
            ensemble_features = torch.stack(adapted_teacher_features, dim=0).mean(dim=0)
            return ensemble_features
    
    def _extract_student_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from student model (penultimate layer)"""
        return self._extract_features_from_model(self.student_model, x)
    
    def _extract_features_from_model(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer of a single model"""
        model.eval()
        with torch.no_grad():
            # Handle custom model wrappers that have a 'model' attribute
            actual_model = model.model if hasattr(model, 'model') else model
            
            # Most models have a structure like: features -> pool -> classifier/fc
            # We want the features before the final classifier
            
            if hasattr(actual_model, 'features') and hasattr(actual_model, 'classifier'):
                # DenseNet, VGG-style models
                features = actual_model.features(x)
                if hasattr(actual_model, 'norm5'):  # DenseNet
                    features = F.relu(actual_model.norm5(features), inplace=True)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                
            elif hasattr(actual_model, 'features') and hasattr(actual_model, 'fc'):
                # EfficientNet-style models
                features = actual_model.features(x)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                
            elif hasattr(actual_model, 'conv1') and hasattr(actual_model, 'stage2') and hasattr(actual_model, 'conv5'):
                # ShuffleNetV2-style models
                x = actual_model.conv1(x)
                x = actual_model.maxpool(x)
                x = actual_model.stage2(x)
                x = actual_model.stage3(x)
                x = actual_model.stage4(x)
                x = actual_model.conv5(x)
                features = x.mean([2, 3])  # Global average pooling (same as ShuffleNet implementation)
                
            elif hasattr(actual_model, 'conv1') and hasattr(actual_model, 'bn1') and hasattr(actual_model, 'fc'):
                # ResNet-style models - forward until before final fc layer
                x = actual_model.conv1(x)
                x = actual_model.bn1(x)
                x = actual_model.relu(x)
                x = actual_model.maxpool(x)
                
                x = actual_model.layer1(x)
                x = actual_model.layer2(x)
                x = actual_model.layer3(x)
                x = actual_model.layer4(x)
                
                x = actual_model.avgpool(x)
                features = torch.flatten(x, 1)
                
            else:
                # Generic fallback - forward through all layers except the last one
                modules = list(actual_model.children())
                if len(modules) > 0:
                    # Remove the last layer (usually classifier)
                    feature_extractor = nn.Sequential(*modules[:-1])
                    features = feature_extractor(x)
                    
                    # Apply global average pooling if features are still spatial
                    if len(features.shape) > 2:
                        features = F.adaptive_avg_pool2d(features, (1, 1))
                        features = torch.flatten(features, 1)
                else:
                    # Last resort: just use a dummy feature vector
                    features = torch.zeros(x.shape[0], 512, device=x.device)
            
            return features
    
    def _get_teacher_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble teacher logits using soft voting"""
        teacher_probs = []
        
        for model in self.teacher_models:
            if "InceptionV3" in type(model).__name__:
                model.train()
                logits, _ = model(x)
                model.eval()
            else:
                logits = model(x)
            
            # Apply teacher temperature
            if self.use_temp_scaling:
                probs = F.softmax(logits / self.teacher_temp, dim=1)
            else:
                probs = F.softmax(logits / self.temperature, dim=1)
            
            teacher_probs.append(probs)
        
        # Average probabilities (soft voting)
        ensemble_probs = torch.stack(teacher_probs, dim=0).mean(dim=0)
        
        # Convert back to logits for KL divergence
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        ensemble_logits = torch.log(ensemble_probs + epsilon)
        
        return ensemble_logits
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass with knowledge distillation
        
        Args:
            x: Input tensor
            targets: Ground truth labels (optional, needed for loss computation)
            
        Returns:
            Dictionary containing logits, features, and losses
        """
        # Get student outputs
        if "InceptionV3" in type(self.student_model).__name__:
            self.student_model.train()
            student_logits, aux_logits = self.student_model(x)
            self.student_model.eval()
        else:
            student_logits = self.student_model(x)
            aux_logits = None
        
        outputs = {
            'student_logits': student_logits,
            'aux_logits': aux_logits
        }
        
        if targets is not None:
            # Compute classification loss
            ce_loss = F.cross_entropy(student_logits, targets)
            if aux_logits is not None:
                aux_loss = F.cross_entropy(aux_logits, targets)
                ce_loss = ce_loss + 0.4 * aux_loss
            
            # Compute logit distillation loss
            with torch.no_grad():
                teacher_logits = self._get_teacher_logits(x)
            
            if self.use_temp_scaling:
                # Advanced temperature scaling
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / self.student_temp, dim=1),
                    F.softmax(teacher_logits / self.teacher_temp, dim=1),
                    reduction='batchmean'
                ) * (self.teacher_temp * self.student_temp)
            else:
                # Basic temperature scaling
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=1),
                    F.softmax(teacher_logits / self.temperature, dim=1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
            
            total_loss = self.alpha * ce_loss + self.beta * kd_loss
            
            # Feature distillation loss
            feature_loss = torch.tensor(0.0, device=self.device)
            if self.use_feature_distillation:
                with torch.no_grad():
                    teacher_features = self._extract_teacher_features(x)
                
                student_features = self._extract_student_features(x)
                
                # Apply adaptation layer
                adapted_student_features = self.feature_adaptation(student_features)
                
                # L2 distance loss
                feature_loss = F.mse_loss(adapted_student_features, teacher_features)
                total_loss += self.gamma * feature_loss
            
            outputs.update({
                'ce_loss': ce_loss,
                'kd_loss': kd_loss,
                'feature_loss': feature_loss,
                'total_loss': total_loss
            })
        
        return outputs
    
    def get_student_model(self) -> nn.Module:
        """Get the student model for inference"""
        return self.student_model
    
    def get_config(self) -> Dict:
        """Get configuration information"""
        return {
            'teacher_models': self.teacher_model_names,
            'student_model': self.student_model_name,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'temperature': self.temperature,
            'teacher_temp': self.teacher_temp,
            'student_temp': self.student_temp,
            'use_feature_distillation': self.use_feature_distillation,
            'use_temp_scaling': self.use_temp_scaling
        } 