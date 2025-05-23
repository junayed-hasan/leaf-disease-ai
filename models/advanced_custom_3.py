"""
Advanced Custom CNN 3: GhostNet-inspired with Ghost Convolutions and Lightweight Attention
Target: ~1.2M parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class GhostModule(nn.Module):
    """Ghost Module for efficient feature generation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, 
                     kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, 
                     groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class LightweightAttention(nn.Module):
    """Lightweight attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class GhostBottleneck(nn.Module):
    """Ghost bottleneck block"""
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, stride, use_se=False):
        super().__init__()
        assert stride in [1, 2]
        
        self.conv = nn.Sequential(
            # Point-wise expansion
            GhostModule(in_channels, hidden_dim, kernel_size=1),
            
            # Depth-wise convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False) if stride > 1 else nn.Identity(),
            nn.BatchNorm2d(hidden_dim) if stride > 1 else nn.Identity(),
            
            # Squeeze-and-excitation
            LightweightAttention(hidden_dim) if use_se else nn.Identity(),
            
            # Point-wise projection
            GhostModule(hidden_dim, out_channels, kernel_size=1)
        )
        
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                         kernel_size//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class AdvancedCustom3(BaseModel):
    """Advanced Custom CNN 3 with Ghost convolutions and lightweight attention"""
    
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__(num_classes, pretrained)
        
        # Building first layer
        output_channel = 16
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        input_channel = output_channel
        
        # Building inverted residual blocks
        stages = []
        
        # Stage 1
        hidden_channel = int(input_channel * 1)
        output_channel = 16
        stages.append(GhostBottleneck(input_channel, hidden_channel, output_channel, 3, 1))
        input_channel = output_channel
        
        # Stage 2
        for i in range(2):
            hidden_channel = int(input_channel * 4)
            output_channel = 24
            stride = 2 if i == 0 else 1
            stages.append(GhostBottleneck(input_channel, hidden_channel, output_channel, 3, stride))
            input_channel = output_channel
        
        # Stage 3
        for i in range(3):
            hidden_channel = int(input_channel * 4)
            output_channel = 40
            stride = 2 if i == 0 else 1
            use_se = True
            stages.append(GhostBottleneck(input_channel, hidden_channel, output_channel, 5, stride, use_se))
            input_channel = output_channel
        
        # Stage 4
        for i in range(4):
            hidden_channel = int(input_channel * 6)
            output_channel = 80 if i < 2 else 112
            stride = 2 if i == 0 else 1
            use_se = True if i >= 2 else False
            stages.append(GhostBottleneck(input_channel, hidden_channel, output_channel, 3, stride, use_se))
            input_channel = output_channel
        
        # Stage 5
        for i in range(3):
            hidden_channel = int(input_channel * 6)
            output_channel = 160
            stride = 2 if i == 0 else 1
            use_se = True
            stages.append(GhostBottleneck(input_channel, hidden_channel, output_channel, 3, stride, use_se))
            input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)
        
        # Building last several layers
        output_channel = 320
        self.conv_head = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(output_channel, num_classes)
        
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
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    @property
    def model_name(self) -> str:
        return "AdvancedCustom3" 