"""
ResNet101 model implementation
"""
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from .base_model import BaseModel

class ResNet101Model(BaseModel):
    """ResNet101 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        self.model = resnet101(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "ResNet101"