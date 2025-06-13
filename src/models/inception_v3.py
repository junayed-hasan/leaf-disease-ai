"""
InceptionV3 model implementation
"""
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from .base_model import BaseModel

class InceptionV3Model(BaseModel):
    """InceptionV3 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        self.model = inception_v3(weights=weights)
        # Modify both main classifier (fc) and auxiliary classifier (AuxLogits.fc)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if hasattr(self.model, 'AuxLogits'):
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)
        
    def forward(self, x):
        # During training, InceptionV3 returns both main output and auxiliary output
        # During inference, it returns only the main output
        if self.training and hasattr(self.model, 'AuxLogits'):
            main_output, aux_output = self.model(x)
            return main_output, aux_output
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "InceptionV3"