#!/usr/bin/env python3
"""
Display Class Mapping for Combined Dataset
Shows the mapping between class names and their corresponding numbers/indices
"""

import json
from pathlib import Path

def load_original_class_mapping():
    """Load the original combined dataset class mapping"""
    
    # Try to find a recent class mapping file
    class_mapping_files = list(Path("outputs").rglob("class_mapping.json"))
    
    # Look for a combined dataset class mapping
    for mapping_file in class_mapping_files:
        if "combined" in str(mapping_file):
            print(f"Loading class mapping from: {mapping_file}")
            with open(mapping_file, 'r') as f:
                return json.load(f)
    
    # If no combined mapping found, use the first available
    if class_mapping_files:
        print(f"Using class mapping from: {class_mapping_files[0]}")
        with open(class_mapping_files[0], 'r') as f:
            return json.load(f)
    
    # Fallback to default mapping (based on what we found in the code)
    print("Using default class mapping")
    return {
        "Tomato___Bacterial_spot": 0,
        "Tomato___Early_blight": 1,
        "Tomato___Late_blight": 2,
        "Tomato___Leaf_Miner": 3,
        "Tomato___Leaf_Mold": 4,
        "Tomato___Magnesium_Deficiency": 5,
        "Tomato___Nitrogen_Deficiency": 6,
        "Tomato___Pottassium_Deficiency": 7,
        "Tomato___Septoria_leaf_spot": 8,
        "Tomato___Spider_mites Two-spotted_spider_mite": 9,
        "Tomato___Spotted_Wilt_Virus": 10,
        "Tomato___Target_Spot": 11,
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 12,
        "Tomato___Tomato_mosaic_virus": 13,
        "Tomato___healthy": 14
    }

def display_class_mapping():
    """Display the class mapping in a clear format"""
    
    print("🔢 COMBINED DATASET CLASS MAPPING")
    print("=" * 80)
    
    # Load the class mapping
    class_mapping = load_original_class_mapping()
    
    print(f"\nTotal Classes: {len(class_mapping)}")
    print("\nClass Name → Number Mapping:")
    print("-" * 80)
    print(f"{'Number':<6} {'Class Name':<50} {'Short Name'}")
    print("-" * 80)
    
    # Sort by class number
    sorted_mapping = sorted(class_mapping.items(), key=lambda x: x[1])
    
    for class_name, class_number in sorted_mapping:
        # Create a shorter version of the class name for readability
        short_name = class_name.replace("Tomato___", "").replace("_", " ")
        print(f"{class_number:<6} {class_name:<50} {short_name}")
    
    print("-" * 80)
    
    # Also show reverse mapping (number to class)
    print(f"\nNumber → Class Name Mapping:")
    print("-" * 80)
    print("# For use in code (number to class name):")
    print("idx_to_class = {")
    
    for class_name, class_number in sorted_mapping:
        print(f"    {class_number}: '{class_name}',")
    
    print("}")
    
    # Show Python dictionary format
    print(f"\n# For use in code (class name to number):")
    print("class_to_idx = {")
    
    for class_name, class_number in sorted_mapping:
        print(f"    '{class_name}': {class_number},")
    
    print("}")
    
    # Save to file
    output_dir = Path("outputs") / "class_mappings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(output_dir / "combined_class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=4)
    
    # Save reverse mapping
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    with open(output_dir / "combined_reverse_mapping.json", "w") as f:
        json.dump(reverse_mapping, f, indent=4)
    
    # Save as Python file
    with open(output_dir / "class_mappings.py", "w") as f:
        f.write("# Combined Dataset Class Mappings\n\n")
        f.write("# Class name to index mapping\n")
        f.write("CLASS_TO_IDX = {\n")
        for class_name, class_number in sorted_mapping:
            f.write(f"    '{class_name}': {class_number},\n")
        f.write("}\n\n")
        
        f.write("# Index to class name mapping\n")
        f.write("IDX_TO_CLASS = {\n")
        for class_name, class_number in sorted_mapping:
            f.write(f"    {class_number}: '{class_name}',\n")
        f.write("}\n\n")
        
        f.write("# Total number of classes\n")
        f.write(f"NUM_CLASSES = {len(class_mapping)}\n")
    
    print(f"\n💾 Class mappings saved to: {output_dir}")
    print(f"   - combined_class_mapping.json")
    print(f"   - combined_reverse_mapping.json")
    print(f"   - class_mappings.py")

def main():
    """Main function"""
    display_class_mapping()
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 