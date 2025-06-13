"""
ShuffleNet V2 (0.5x) model implementation
"""
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from .base_model import BaseModel

class ShuffleNetV2Model(BaseModel):
    """ShuffleNet V2 (0.5x) model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        self.model = shufflenet_v2_x0_5(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
    @property
    def model_name(self) -> str:
        return "ShuffleNetV2" 