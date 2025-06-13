"""
Swin Large model implementation using timm
"""
import torch
import torch.nn as nn
from timm import create_model
from .base_model import BaseModel

class SwinLargeModel(BaseModel):
    """Swin Large model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        # Initialize the model using timm
        model_name = 'swinv2_large_window12_192_22k'
        print(f"Initializing Swin Large model with name: {model_name}")
        
        try:
            self.model = create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            print("Successfully created Swin Large model")
        except Exception as e:
            print(f"Error creating Swin Large model: {str(e)}")
            raise
    
    def forward(self, x):
        # Add input size check for debugging
        if x.shape[-2:] != (192, 192):
            print(f"Warning: Input size {x.shape[-2:]} doesn't match expected size (192, 192)")
        return self.model(x)
    
    @property
    def model_name(self) -> str:
        return "Swin-Large"