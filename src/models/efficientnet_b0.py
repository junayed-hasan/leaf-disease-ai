"""
EfficientNet-B0 model implementation
"""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from .base_model import BaseModel

class EfficientNetB0Model(BaseModel):
    """EfficientNet-B0 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "EfficientNet-B0"