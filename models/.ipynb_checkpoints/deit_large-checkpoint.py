"""
DeiT Large model implementation using timm
"""
import torch
import torch.nn as nn
from timm import create_model
from .base_model import BaseModel

class DeiTLargeModel(BaseModel):
    """DeiT Large model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        # Initialize the model using timm
        self.model = create_model(
            'deit3_large_patch16_224',  # DeiT-Large model name in timm
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)
    
    @property
    def model_name(self) -> str:
        return "DeiT-Large"