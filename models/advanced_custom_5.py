"""
Advanced Custom CNN 5: Hybrid CNN-Transformer with Efficient Self-Attention
Target: ~1.5M parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel

class PatchEmbedding(nn.Module):
    """Image to Patch Embedding using Conv2d"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x

class EfficientAttention(nn.Module):
    """Efficient self-attention with linear complexity"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    """Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer block with efficient attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class CNNBackbone(nn.Module):
    """Lightweight CNN backbone for feature extraction"""
    def __init__(self, out_channels=96):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.features(x)

class AdvancedCustom5(BaseModel):
    """Advanced Custom CNN 5 with lightweight hybrid CNN-Transformer architecture"""
    
    def __init__(self, num_classes: int, pretrained: bool = False, img_size=224, embed_dims=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1]):
        super().__init__(num_classes, pretrained)
        
        self.num_classes = num_classes
        self.depths = depths
        
        # Lightweight CNN backbone for initial feature extraction
        self.cnn_backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, embed_dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            
            # Stage 2  
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )
        
        # Simplified patch embeddings - just conv layers for downsampling
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        
        # Calculate feature map sizes: 224->112->56, then 56->28->14->7
        sizes = [56, 28, 14, 7]
        
        for i in range(4):
            if i == 0:
                # First stage uses CNN features directly
                patch_embed = nn.Identity()
                pos_embed = nn.Parameter(torch.zeros(1, sizes[i] * sizes[i], embed_dims[i]))
            else:
                # Subsequent stages use conv for downsampling
                patch_embed = nn.Conv2d(embed_dims[i-1], embed_dims[i], kernel_size=3, stride=2, padding=1)
                pos_embed = nn.Parameter(torch.zeros(1, sizes[i] * sizes[i], embed_dims[i]))
            
            self.patch_embeds.append(patch_embed)
            self.pos_embeds.append(pos_embed)
            self.pos_drops.append(nn.Dropout(p=drop_rate))
        
        # Lightweight transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks = nn.ModuleList()
        for i in range(4):
            blk = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            self.blocks.append(blk)
            cur += depths[i]
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dims[3])
        self.head = nn.Linear(embed_dims[3], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        for pos_embed in self.pos_embeds:
            nn.init.trunc_normal_(pos_embed, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x):
        B = x.shape[0]
        sizes = [56, 28, 14, 7]
        
        # CNN backbone
        x = self.cnn_backbone(x)  # B, 32, 56, 56
        
        # Transformer stages
        for i in range(4):
            if i == 0:
                # First stage: flatten CNN features
                x = x.flatten(2).transpose(1, 2)  # B, 56*56, 32
                H, W = sizes[i], sizes[i]
            else:
                # Subsequent stages: apply conv downsampling then flatten
                x = x.permute(0, 2, 1).reshape(B, -1, H, W)  # Reshape back to spatial
                x = self.patch_embeds[i](x)  # Apply conv downsampling
                x = x.flatten(2).transpose(1, 2)  # Flatten again
                H, W = sizes[i], sizes[i]
            
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            
            # Apply transformer blocks
            for blk in self.blocks[i]:
                x = blk(x, H, W)
        
        return self.norm(x.mean(dim=1))  # Global average pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    @property
    def model_name(self) -> str:
        return "AdvancedCustom5" 