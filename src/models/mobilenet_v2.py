"""
MobileNetV2 model implementation
"""
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from .base_model import BaseModel

class MobileNetV2Model(BaseModel):
    """MobileNetV2 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v2(weights=weights)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "MobileNetV2"