"""
EfficientNet-B4 model implementation
"""
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from .base_model import BaseModel

class EfficientNetB4Model(BaseModel):
    """EfficientNet-B4 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b4(weights=weights)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "EfficientNet-B4"