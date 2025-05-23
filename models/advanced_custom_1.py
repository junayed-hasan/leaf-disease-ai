"""
Advanced Custom CNN 1: EfficientNet-inspired with Inverted Residuals and SE blocks
Target: ~0.8M parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    """Inverted residual block (MobileNetV2-style) with SE"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6, use_se=True):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, 
                     groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        ])
        
        # Squeeze-Excitation
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))
        
        # Projection phase
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out

class AdvancedCustom1(BaseModel):
    """Advanced Custom CNN 1 with inverted residuals and SE blocks"""
    
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__(num_classes, pretrained)
        
        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks configuration: [expand_ratio, channels, repeats, stride]
        configs = [
            [1, 16, 1, 1],    # 112x112
            [6, 24, 2, 2],    # 56x56
            [6, 40, 3, 2],    # 28x28
            [6, 80, 3, 2],    # 14x14
            [6, 112, 2, 1],   # 14x14
            [6, 192, 1, 2],   # 7x7
        ]
        
        layers = []
        in_channels = 32
        
        for expand_ratio, out_channels, repeats, stride in configs:
            for i in range(repeats):
                layers.append(InvertedResidualBlock(
                    in_channels, out_channels, 
                    stride if i == 0 else 1, 
                    expand_ratio, 
                    use_se=True
                ))
                in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Final layers
        self.conv_final = nn.Sequential(
            nn.Conv2d(192, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.conv_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    @property
    def model_name(self) -> str:
        return "AdvancedCustom1" 