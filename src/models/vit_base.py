"""
ViT-Base (B/16) model implementation
"""
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from .base_model import BaseModel

class ViTBaseModel(BaseModel):
    """ViT-Base (B/16) model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = vit_b_16(weights=weights)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "ViT-Base"
