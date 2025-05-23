"""
ViT-Large (L/32) model implementation
"""
import torch.nn as nn
from torchvision.models import vit_l_32, ViT_L_32_Weights
from .base_model import BaseModel

class ViTLargeModel(BaseModel):
    """ViT-Large (L/32) model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = ViT_L_32_Weights.DEFAULT if pretrained else None
        self.model = vit_l_32(weights=weights)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "ViT-Large"
