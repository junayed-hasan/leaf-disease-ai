"""
Swin-Tiny model implementation
"""
import torch.nn as nn
from timm import create_model
from .base_model import BaseModel

class SwinTinyModel(BaseModel):
    """Swin-Tiny model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        # Use `create_model` to load Swin-Tiny
        self.model = create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        # Replace the classification head to match the number of classes
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "Swin-Tiny"
