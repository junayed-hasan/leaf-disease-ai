#!/usr/bin/env python3
"""
Script to test the new advanced custom models and check their parameter counts
"""
import torch
from models.model_factory import ModelFactory

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_model(model_name):
    try:
        print(f"\n--- Testing {model_name} ---")
        model = ModelFactory.get_model(model_name, num_classes=15, pretrained=False)
        
        # Count parameters
        params = count_parameters(model)
        print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Check if parameters are reasonable (< 2M)
        if params < 2_000_000:
            print("✅ Parameter count within limit")
        else:
            print("❌ Parameter count exceeds 2M limit")
        
        return True, params
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False, 0

def main():
    print("Testing Advanced Custom Models")
    print("="*50)
    
    # List of new advanced models to test
    advanced_models = [
        "advanced_custom_1",
        "advanced_custom_2", 
        "advanced_custom_3",
        "advanced_custom_4",
        "advanced_custom_5"
    ]
    
    results = {}
    
    for model_name in advanced_models:
        success, params = test_model(model_name)
        results[model_name] = (success, params)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print(f"{'Model':<20} {'Parameters':<15} {'Status':<10}")
    print("-" * 45)
    
    for model_name, (success, params) in results.items():
        status = "✅ OK" if success and params < 2_000_000 else "❌ FAIL"
        params_str = f"{params:,} ({params/1e6:.2f}M)" if success else "N/A"
        print(f"{model_name:<20} {params_str:<15} {status}")
    
    # Compare with ShuffleNet V2 target
    print(f"\nTarget to beat: ShuffleNet V2 with 92.12% F1 and 0.36M parameters")
    print("Goal: Achieve >92.12% F1 with <2M parameters")

if __name__ == "__main__":
    main() 