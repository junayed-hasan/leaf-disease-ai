"""
DenseNet121 model implementation
"""
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from .base_model import BaseModel

class DenseNet121Model(BaseModel):
    """DenseNet121 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = densenet121(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "DenseNet121"