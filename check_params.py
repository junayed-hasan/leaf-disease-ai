#!/usr/bin/env python3
"""
Script to check parameter counts of pretrained small models
"""
import torch
from torchvision.models import mobilenet_v2, squeezenet1_1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Check parameter counts
models = {
    'MobileNetV2': mobilenet_v2(pretrained=False),
    'SqueezeNet1.1': squeezenet1_1(pretrained=False),
}

print('Parameter counts of pretrained small models:')
for name, model in models.items():
    params = count_parameters(model)
    print(f'{name}: {params:,} parameters ({params/1e6:.2f}M)')

# Also check our custom models
try:
    from models.shufflenet_v2 import ShuffleNetV2Model
    shufflenet = ShuffleNetV2Model(15, pretrained=False)
    params = count_parameters(shufflenet.model)
    print(f'ShuffleNetV2: {params:,} parameters ({params/1e6:.2f}M)')
except Exception as e:
    print(f'Could not load ShuffleNetV2: {e}')

try:
    from models.mobilenet_v3_small import MobileNetV3SmallModel
    mobilenet_v3 = MobileNetV3SmallModel(15, pretrained=False)
    params = count_parameters(mobilenet_v3.model)
    print(f'MobileNetV3Small: {params:,} parameters ({params/1e6:.2f}M)')
except Exception as e:
    print(f'Could not load MobileNetV3Small: {e}') 