"""
Base model class that all model architectures should inherit from
"""
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model architecture"""
        pass