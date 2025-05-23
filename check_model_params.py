#!/usr/bin/env python3
"""
Script to check parameter counts of pretrained small models
"""
import torch
from models.shufflenet_v2 import ShuffleNetV2Model
from models.mobilenet_v3_small import MobileNetV3SmallModel
from models.squeezenet import SqueezeNetModel
from models.mobilenet_v2 import MobileNetV2Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Check parameter counts
models = [
    ('ShuffleNetV2', ShuffleNetV2Model(15, False)),
    ('MobileNetV3Small', MobileNetV3SmallModel(15, False)), 
    ('SqueezeNet', SqueezeNetModel(15, False)),
    ('MobileNetV2', MobileNetV2Model(15, False))
]

print('Parameter counts of pretrained small models:')
for name, model in models:
    params = count_parameters(model)
    print(f'{name}: {params:,} parameters ({params/1e6:.2f}M)') 