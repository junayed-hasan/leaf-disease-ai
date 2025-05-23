"""
Xception model implementation using timm
"""
import torch.nn as nn
import timm
from .base_model import BaseModel

class XceptionModel(BaseModel):
    """Xception model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.model = timm.create_model('xception', pretrained=pretrained)
        # Modify the classifier
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "Xception"