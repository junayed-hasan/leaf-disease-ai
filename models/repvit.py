"""
RepViT-A0 model implementation
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class RepViTA0Model(BaseModel):
    """RepViT-A0 model for tomato disease classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        try:
            # Try to import timm
            import timm
            
            # Load RepViT-A0 model
            if pretrained:
                self.model = timm.create_model('repvit_a0', pretrained=True)
            else:
                self.model = timm.create_model('repvit_a0', pretrained=False)
                
            # Get the input features dimension for the main head
            if hasattr(self.model.head, 'fc'):
                in_features = self.model.head.fc.in_features
                # Replace the classifier head
                self.model.head.fc = nn.Linear(in_features, num_classes)
            else:
                in_features = self.model.head.in_features
                # Replace the classifier head
                self.model.head = nn.Linear(in_features, num_classes)
            
            # Replace the distillation head if it exists (similar to EfficientFormer)
            if hasattr(self.model, 'dist_head') and self.model.dist_head is not None:
                if hasattr(self.model.dist_head, 'fc'):
                    self.model.dist_head.fc = nn.Linear(in_features, num_classes)
                else:
                    self.model.dist_head = nn.Linear(in_features, num_classes)
            
            # Disable distillation during inference if the model supports it
            if hasattr(self.model, 'distillation_training'):
                self.model.distillation_training = False
            
        except ImportError:
            raise ImportError(
                "Please install timm package to use RepViT: pip install timm"
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
                
            # Apply the classification head based on its structure
            if hasattr(self.model.head, 'fc'):
                x = self.model.head.fc(x)
            else:
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
                # Flatten if needed
                features = torch.flatten(features, 1) if features.dim() > 2 else features
                # Apply head based on structure
                if hasattr(self.model.head, 'fc'):
                    return self.model.head.fc(features)
                else:
                    return self.model.head(features)
        
    @property
    def model_name(self) -> str:
        return "RepViTA0" 