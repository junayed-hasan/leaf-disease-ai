"""
EfficientFormer-L1 model implementation
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class EfficientFormerL1Model(BaseModel):
    """EfficientFormer-L1 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        try:
            # Try to import timm
            import timm
            
            # Load EfficientFormer-L1 model
            if pretrained:
                self.model = timm.create_model('efficientformer_l1', pretrained=True)
            else:
                self.model = timm.create_model('efficientformer_l1', pretrained=False)
                
            # Replace both the main classifier head and distillation head
            # EfficientFormer uses a knowledge distillation architecture
            
            # Get the input features dimension
            in_features = self.model.head.in_features
            
            # Replace the main classifier head
            self.model.head = nn.Linear(in_features, num_classes)
            
            # Replace the distillation head if it exists
            if hasattr(self.model, 'dist_head') and self.model.dist_head is not None:
                self.model.dist_head = nn.Linear(in_features, num_classes)
            
            # Disable distillation during inference if the model supports it
            if hasattr(self.model, 'distillation_training'):
                self.model.distillation_training = False
            
        except ImportError:
            raise ImportError(
                "Please install timm package to use EfficientFormer: pip install timm"
            )
        
    def forward(self, x):
        """
        Custom forward method that correctly handles the model output regardless of dimensions
        """
        try:
            # The safest approach - let the model handle the forward pass directly
            # but modify the outputs to fit our needs
            features = self.model.forward_features(x)
            
            # Check if features is already a 2D tensor (batch_size, features)
            if features.dim() == 2:
                x = features
            # Handle 3D tensor (batch_size, seq_len, features) - typical for transformer outputs
            elif features.dim() == 3:
                # Get the class token or average across sequence length
                if hasattr(self.model, 'cls_token') and self.model.cls_token is not None:
                    # Use class token if it exists (first token)
                    x = features[:, 0]
                else:
                    # Average across sequence length
                    x = torch.mean(features, dim=1)
            # Handle 4D tensor (batch_size, channels, height, width) - typical for CNN outputs
            elif features.dim() == 4:
                # Global average pooling
                x = torch.mean(features, dim=(2, 3))
            else:
                raise ValueError(f"Unexpected feature dimension: {features.dim()}")
                
            # Apply the classification head
            x = self.model.head(x)
            return x
            
        except Exception as e:
            # Fallback to a very simple approach if anything goes wrong
            print(f"Warning: Using fallback forward method due to error: {str(e)}")
            
            # Let's use the model's forward method but catch and fix any errors
            try:
                # Try the model's own forward method but with a try-except to catch errors
                return self.model(x)
            except Exception:
                # If that fails, extract features and directly apply head
                features = self.model.forward_features(x)
                if features.dim() > 2:
                    # If features have more than 2 dimensions, flatten them
                    features = torch.flatten(features, 1) if features.dim() > 2 else features
                return self.model.head(features)
        
    @property
    def model_name(self) -> str:
        return "EfficientFormerL1" 