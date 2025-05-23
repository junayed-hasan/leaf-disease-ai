"""
Custom CNN with approximately 500,000 parameters
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class ResidualBlock(nn.Module):
    """Simple residual block with same input/output channels"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class CustomCNN500K(BaseModel):
    """Custom CNN with approx. 500K parameters for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__(num_classes, pretrained)
        
        # Define a CNN architecture with ~500K parameters
        # Input: 3 x 224 x 224
        self.features = nn.Sequential(
            # Layer 1: 3 -> 64 channels, 7x7 kernel
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Size: 64 x 56 x 56
            
            # Layer 2: Residual block (64 channels)
            ResidualBlock(64),
            # Size: 64 x 56 x 56
            
            # Layer 3: 64 -> 128 channels, 3x3 kernel
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Size: 128 x 28 x 28
            
            # Layer 4: Residual block (128 channels)
            ResidualBlock(128),
            # Size: 128 x 28 x 28
            
            # Layer 5: 128 -> 256 channels, 3x3 kernel
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Size: 256 x 14 x 14
            
            # Layer 6: Residual block (256 channels)
            ResidualBlock(256),
            # Size: 256 x 14 x 14
            
            # Layer 7: 256 -> 384 channels, 3x3 kernel
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Size: 384 x 7 x 7
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
            # Size: 384 x 1 x 1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
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
        return "CustomCNN500K" 