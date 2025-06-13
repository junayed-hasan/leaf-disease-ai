"""
Custom CNN with approximately 100,000 parameters
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class CustomCNN100K(BaseModel):
    """Custom CNN with approx. 100K parameters for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__(num_classes, pretrained)
        
        # Define a CNN architecture with ~100K parameters
        # Input: 3 x 224 x 224
        self.features = nn.Sequential(
            # Layer 1: 3 -> 32 channels, 7x7 kernel
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Size: 32 x 56 x 56
            
            # Layer 2: 32 -> 64 channels, 3x3 kernel
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Size: 64 x 28 x 28
            
            # Layer 3: 64 -> 96 channels, 3x3 kernel
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Size: 96 x 14 x 14
            
            # Layer 4: 96 -> 128 channels, 3x3 kernel
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Size: 128 x 7 x 7
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
            # Size: 128 x 1 x 1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
    @property
    def model_name(self) -> str:
        return "CustomCNN100K" 