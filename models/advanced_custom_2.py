"""
Advanced Custom CNN 2: ShuffleNet-inspired with Channel Shuffle and Grouped Convolutions
Target: ~0.6M parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

def channel_shuffle(x, groups):
    """Channel shuffle operation"""
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    
    # Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    
    # Transpose
    x = torch.transpose(x, 1, 2).contiguous()
    
    # Flatten
    x = x.view(batch_size, -1, height, width)
    
    return x

class ShuffleUnit(nn.Module):
    """Simplified ShuffleNet unit without problematic concatenation"""
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        self.stride = stride
        
        # Simplified: no concatenation, just residual connection
        mid_channels = out_channels // 2
        
        # Find compatible groups for conv1
        groups1 = min(groups, in_channels, mid_channels)
        while (in_channels % groups1 != 0 or mid_channels % groups1 != 0) and groups1 > 1:
            groups1 -= 1
        
        # Find compatible groups for conv3
        groups3 = min(groups, mid_channels, out_channels)
        while (mid_channels % groups3 != 0 or out_channels % groups3 != 0) and groups3 > 1:
            groups3 -= 1
        
        self.groups = groups3
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, 
                     groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups3, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        out = self.conv1(x)
        out = channel_shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = out + shortcut
        out = F.relu(out, inplace=True)
        return out

class AdvancedCustom2(BaseModel):
    """Advanced Custom CNN 2 with ShuffleNet-inspired architecture"""
    
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__(num_classes, pretrained)
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # ShuffleNet stages - Simplified without concatenation
        # Stage 2: 24 -> 96
        self.stage2 = nn.Sequential(
            ShuffleUnit(24, 96, stride=2, groups=1),
            ShuffleUnit(96, 96, stride=1, groups=4),
            ShuffleUnit(96, 96, stride=1, groups=4),
            ShuffleUnit(96, 96, stride=1, groups=4)
        )
        
        # Stage 3: 96 -> 192
        self.stage3 = nn.Sequential(
            ShuffleUnit(96, 192, stride=2, groups=4),
            ShuffleUnit(192, 192, stride=1, groups=8),
            ShuffleUnit(192, 192, stride=1, groups=8),
            ShuffleUnit(192, 192, stride=1, groups=8)
        )
        
        # Stage 4: 192 -> 384
        self.stage4 = nn.Sequential(
            ShuffleUnit(192, 384, stride=2, groups=8),
            ShuffleUnit(384, 384, stride=1, groups=8),
            ShuffleUnit(384, 384, stride=1, groups=8)
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(384, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'pointwise' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
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
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    @property
    def model_name(self) -> str:
        return "AdvancedCustom2"