"""
Efficient ViT model implementation using EfficientFormer
"""
import torch
import torch.nn as nn
from timm import create_model
from .base_model import BaseModel

class EfficientViTModel(BaseModel):
    """EfficientFormer model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        # Initialize the EfficientFormer model
        self.model = create_model(
            'efficientformer_l1',  # Using EfficientFormer-L1 architecture
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # If you need to modify the head
        if hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    @property
    def model_name(self) -> str:
        return "EfficientViT"