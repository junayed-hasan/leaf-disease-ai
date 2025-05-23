"""
SqueezeNet 1.1 model implementation
"""
import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from .base_model import BaseModel

class SqueezeNetModel(BaseModel):
    """SqueezeNet 1.1 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        self.model = squeezenet1_1(weights=weights)
        
        # Replace the classifier
        self.model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=1, stride=1
        )
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "SqueezeNet" 