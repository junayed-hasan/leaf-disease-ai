"""
VGG16 model implementation
"""
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from .base_model import BaseModel

class VGG16Model(BaseModel):
    """VGG16 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = vgg16(weights=weights)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "VGG16"