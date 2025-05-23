"""
Custom CNN with approximately 2 million parameters
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class BottleneckBlock(nn.Module):
    """Bottleneck block (similar to ResNet bottleneck)"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        bottleneck_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 
                             kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class CustomCNN2M(BaseModel):
    """Custom CNN with approx. 2M parameters for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__(num_classes, pretrained)
        
        # Define a CNN architecture with ~2M parameters
        # Input: 3 x 224 x 224
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Size: 64 x 56 x 56
        
        # Bottleneck blocks
        self.layer1 = nn.Sequential(
            BottleneckBlock(64, 256),
            BottleneckBlock(256, 256)
        )
        # Size: 256 x 56 x 56
        
        self.layer2 = nn.Sequential(
            BottleneckBlock(256, 512, stride=2),
            BottleneckBlock(512, 512)
        )
        # Size: 512 x 28 x 28
        
        self.layer3 = nn.Sequential(
            BottleneckBlock(512, 768, stride=2),
            BottleneckBlock(768, 768)
        )
        # Size: 768 x 14 x 14
        
        self.layer4 = nn.Sequential(
            BottleneckBlock(768, 1024, stride=2)
        )
        # Size: 1024 x 7 x 7
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Size: 1024 x 1 x 1
        
        # Classifier with attention
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
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
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Bottleneck layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply attention mechanism
        attn = self.attention(x)
        x = x * attn
        
        # Classifier
        x = self.classifier(x)
        
        return x
        
    @property
    def model_name(self) -> str:
        return "CustomCNN2M" 