"""
Custom CNN with approximately 10,000 parameters
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class CustomCNN10K(BaseModel):
    """Custom CNN with approx. 10K parameters for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__(num_classes, pretrained)
        
        # Define a compact CNN architecture with ~10K parameters
        # Input: 3 x 224 x 224
        self.features = nn.Sequential(
            # Layer 1: 3 -> 16 channels, 7x7 kernel
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Size: 16 x 56 x 56
            
            # Layer 2: 16 -> 32 channels, 3x3 kernel with depthwise separable convolution
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),  # Depthwise
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),  # Pointwise
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Size: 32 x 28 x 28
            
            # Layer 3: 32 -> 48 channels, 3x3 kernel with depthwise separable convolution
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),  # Depthwise
            nn.Conv2d(32, 48, kernel_size=1, stride=1, padding=0),  # Pointwise
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Size: 48 x 14 x 14
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
            # Size: 48 x 1 x 1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, num_classes)
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
        return "CustomCNN10K" 