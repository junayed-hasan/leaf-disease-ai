"""
Factory class for creating model instances
"""
from typing import Dict, Type, List
from .base_model import BaseModel
from .resnet50 import ResNet50Model
from .resnet101 import ResNet101Model
from .densenet121 import DenseNet121Model
from .densenet201 import DenseNet201Model
from .vgg16 import VGG16Model
from .vgg19 import VGG19Model
from .inception_v3 import InceptionV3Model
from .xception import XceptionModel
from .mobilenet_v2 import MobileNetV2Model
from .efficientnet_b0 import EfficientNetB0Model
from .efficientnet_b4 import EfficientNetB4Model
from .vit_base import ViTBaseModel
from .vit_large import ViTLargeModel
from .deit_small import DeiTSmallModel
from .deit_base import DeiTBaseModel
from .deit_large import DeiTLargeModel  
from .swin_tiny import SwinTinyModel
from .swin_base import SwinBaseModel
from .swin_large import SwinLargeModel  
from .efficient_vit import EfficientViTModel

# New lightweight models for student models
from .mobilenet_v3_small import MobileNetV3SmallModel
from .shufflenet_v2 import ShuffleNetV2Model
from .squeezenet import SqueezeNetModel
from .efficientformer import EfficientFormerL1Model
from .repvit import RepViTA0Model

# Custom CNN models with varying parameter counts
from .custom_cnn_10k import CustomCNN10K
from .custom_cnn_100k import CustomCNN100K
from .custom_cnn_500k import CustomCNN500K
from .custom_cnn_1m import CustomCNN1M
from .custom_cnn_2m import CustomCNN2M

# Advanced custom models with modern architectures
from .advanced_custom_1 import AdvancedCustom1
from .advanced_custom_2 import AdvancedCustom2
from .advanced_custom_3 import AdvancedCustom3
from .advanced_custom_4 import AdvancedCustom4
from .advanced_custom_5 import AdvancedCustom5

class ModelFactory:
    """Factory class to create model instances"""
    
    _models: Dict[str, Type[BaseModel]] = {
        "resnet50": ResNet50Model,
        "resnet101": ResNet101Model,
        "densenet121": DenseNet121Model,
        "densenet201": DenseNet201Model,
        "vgg16": VGG16Model,
        "vgg19": VGG19Model,
        "inception_v3": InceptionV3Model,
        "xception": XceptionModel,
        "mobilenet_v2": MobileNetV2Model,
        "efficientnet_b0": EfficientNetB0Model,
        "efficientnet_b4": EfficientNetB4Model,
        "vit_base": ViTBaseModel,
        "vit_large": ViTLargeModel,
        "deit_small": DeiTSmallModel,
        "deit_base": DeiTBaseModel,
        "deit_large": DeiTLargeModel,
        "swin_tiny": SwinTinyModel,
        "swin_base": SwinBaseModel,
        "swin_large": SwinLargeModel,
        "efficient_vit": EfficientViTModel,
        
        # New lightweight models
        "mobilenet_v3_small": MobileNetV3SmallModel,
        "shufflenet_v2": ShuffleNetV2Model,
        "squeezenet": SqueezeNetModel,
        "efficientformer_l1": EfficientFormerL1Model,
        "repvit_a0": RepViTA0Model,
        
        # Custom CNN models
        "custom_cnn_10k": CustomCNN10K,
        "custom_cnn_100k": CustomCNN100K,
        "custom_cnn_500k": CustomCNN500K,
        "custom_cnn_1m": CustomCNN1M,
        "custom_cnn_2m": CustomCNN2M,
        
        # Advanced custom models
        "advanced_custom_1": AdvancedCustom1,
        "advanced_custom_2": AdvancedCustom2,
        "advanced_custom_3": AdvancedCustom3,
        "advanced_custom_4": AdvancedCustom4,
        "advanced_custom_5": AdvancedCustom5
    }
    
    @classmethod
    def get_model(cls, model_name: str, num_classes: int, pretrained: bool = True) -> BaseModel:
        """
        Create and return a model instance
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            
        Returns:
            An instance of the specified model
        
        Raises:
            ValueError: If model_name is not recognized
        """
        try:
            model_class = cls._models[model_name.lower()]
            return model_class(num_classes=num_classes, pretrained=pretrained)
        except KeyError:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(cls._models.keys())}"
            )
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return list of available model architectures"""
        return list(cls._models.keys())