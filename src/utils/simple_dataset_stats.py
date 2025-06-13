#!/usr/bin/env python3
"""
Simple Dataset Statistics Analysis
Prints comprehensive statistics for all datasets: plantVillage, tomatoVillage, and combined
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# Import existing modules that we know exist
from config import *

def get_dataset_statistics(dataset_path, dataset_name):
    """Get comprehensive statistics for a dataset"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return {
            'dataset_name': dataset_name,
            'exists': False,
            'error': f"Dataset path not found: {dataset_path}"
        }
    
    stats = {
        'dataset_name': dataset_name,
        'exists': True,
        'dataset_path': str(dataset_path),
        'class_names': [],
        'class_counts': {},
        'total_samples': 0,
        'num_classes': 0
    }
    
    try:
        # Count samples in each class directory
        class_counts = {}
        total_samples = 0
        
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                
                # Count image files in class directory
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                image_files = [
                    f for f in class_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
                
                sample_count = len(image_files)
                class_counts[class_name] = sample_count
                total_samples += sample_count
        
        stats['class_names'] = sorted(class_counts.keys())
        stats['class_counts'] = class_counts
        stats['total_samples'] = total_samples
        stats['num_classes'] = len(class_counts)
        
        # Additional statistics
        if class_counts:
            counts_list = list(class_counts.values())
            stats['min_samples_per_class'] = min(counts_list)
            stats['max_samples_per_class'] = max(counts_list)
            stats['avg_samples_per_class'] = round(sum(counts_list) / len(counts_list), 2)
            stats['class_balance_ratio'] = round(stats['max_samples_per_class'] / stats['min_samples_per_class'], 2)
        
    except Exception as e:
        stats['error'] = str(e)
    
    return stats

def get_split_dataset_statistics(train_dir, val_dir, test_dir, dataset_name):
    """Get statistics for a dataset that's already split into train/val/test"""
    
    stats = {
        'dataset_name': dataset_name,
        'exists': True,
        'dataset_path': f"train: {train_dir}, val: {val_dir}, test: {test_dir}",
        'class_names': [],
        'class_counts': {},
        'total_samples': 0,
        'num_classes': 0,
        'split_counts': {'train': 0, 'val': 0, 'test': 0}
    }
    
    try:
        all_class_counts = {}
        total_samples = 0
        
        # Process each split
        splits = {'train': train_dir, 'val': val_dir, 'test': test_dir}
        
        for split_name, split_path in splits.items():
            split_path = Path(split_path)
            split_total = 0
            
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir() and not class_dir.name.startswith('.'):
                        class_name = class_dir.name
                        
                        # Count image files in class directory
                        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                        image_files = [
                            f for f in class_dir.iterdir() 
                            if f.is_file() and f.suffix.lower() in image_extensions
                        ]
                        
                        sample_count = len(image_files)
                        
                        if class_name not in all_class_counts:
                            all_class_counts[class_name] = 0
                        all_class_counts[class_name] += sample_count
                        
                        split_total += sample_count
                        total_samples += sample_count
            
            stats['split_counts'][split_name] = split_total
        
        stats['class_names'] = sorted(all_class_counts.keys())
        stats['class_counts'] = all_class_counts
        stats['total_samples'] = total_samples
        stats['num_classes'] = len(all_class_counts)
        
        # Additional statistics
        if all_class_counts:
            counts_list = list(all_class_counts.values())
            stats['min_samples_per_class'] = min(counts_list)
            stats['max_samples_per_class'] = max(counts_list)
            stats['avg_samples_per_class'] = round(sum(counts_list) / len(counts_list), 2)
            stats['class_balance_ratio'] = round(stats['max_samples_per_class'] / stats['min_samples_per_class'], 2)
        
    except Exception as e:
        stats['error'] = str(e)
        stats['exists'] = False
    
    return stats

def get_combined_dataset_statistics():
    """Get statistics for the combined dataset"""
    
    stats = {
        'dataset_name': 'combined',
        'exists': True,
        'source': 'plantvillage + tomatovillage combined'
    }
    
    try:
        # Get individual dataset configs
        plantvillage_config = get_dataset_config('plantvillage')
        tomatovillage_config = get_dataset_config('tomatovillage')
        
        # Combined class mapping (based on what we've seen in the project)
        combined_classes = {
            'Tomato___Bacterial_spot': 0,
            'Tomato___Early_blight': 1,
            'Tomato___Late_blight': 2,
            'Tomato___Leaf_Mold': 3,
            'Tomato___Septoria_leaf_spot': 4,
            'Tomato___Spider_mites_Two-spotted_spider_mite': 5,
            'Tomato___Target_Spot': 6,
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 7,
            'Tomato___Tomato_mosaic_virus': 8,
            'Tomato___healthy': 9,
            'Tomato___Leaf_Miner': 10,
            'Tomato___Magnesium_Deficiency': 11,
            'Tomato___Nitrogen_Deficiency': 12,
            'Tomato___Pottassium_Deficiency': 13,
            'Tomato___Spotted_Wilt_Virus': 14
        }
        
        stats['class_names'] = sorted(combined_classes.keys())
        stats['num_classes'] = len(combined_classes)
        
        # Count samples across all source datasets
        total_samples = 0
        class_counts = defaultdict(int)
        
        # Process plantvillage
        plantvillage_path = plantvillage_config['root']
        if plantvillage_path.exists():
            for class_dir in plantvillage_path.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith('.'):
                    class_name = class_dir.name
                    
                    # Map to combined class name
                    combined_class = None
                    if class_name in combined_classes:
                        combined_class = class_name
                    else:
                        # Try to find matching class
                        for combined_name in combined_classes.keys():
                            if (class_name.lower() in combined_name.lower() or 
                                combined_name.lower() in class_name.lower() or
                                class_name.replace('_', ' ').lower() in combined_name.replace('_', ' ').lower()):
                                combined_class = combined_name
                                break
                    
                    if combined_class:
                        # Count image files
                        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                        image_files = [
                            f for f in class_dir.iterdir() 
                            if f.is_file() and f.suffix.lower() in image_extensions
                        ]
                        
                        sample_count = len(image_files)
                        class_counts[combined_class] += sample_count
                        total_samples += sample_count
        
        # Process tomatovillage (all splits)
        tomatovillage_splits = ['train_dir', 'val_dir', 'test_dir']
        for split in tomatovillage_splits:
            if split in tomatovillage_config:
                split_path = tomatovillage_config[split]
                if split_path.exists():
                    for class_dir in split_path.iterdir():
                        if class_dir.is_dir() and not class_dir.name.startswith('.'):
                            class_name = class_dir.name
                            
                            # Map to combined class name
                            combined_class = None
                            if class_name in combined_classes:
                                combined_class = class_name
                            else:
                                # Try to find matching class
                                for combined_name in combined_classes.keys():
                                    if (class_name.lower() in combined_name.lower() or 
                                        combined_name.lower() in class_name.lower() or
                                        class_name.replace('_', ' ').lower() in combined_name.replace('_', ' ').lower()):
                                        combined_class = combined_name
                                        break
                            
                            if combined_class:
                                # Count image files
                                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                                image_files = [
                                    f for f in class_dir.iterdir() 
                                    if f.is_file() and f.suffix.lower() in image_extensions
                                ]
                                
                                sample_count = len(image_files)
                                class_counts[combined_class] += sample_count
                                total_samples += sample_count
        
        stats['class_counts'] = dict(class_counts)
        stats['total_samples'] = total_samples
        
        # Additional statistics
        if class_counts:
            counts_list = list(class_counts.values())
            stats['min_samples_per_class'] = min(counts_list) if counts_list else 0
            stats['max_samples_per_class'] = max(counts_list) if counts_list else 0
            stats['avg_samples_per_class'] = round(sum(counts_list) / len(counts_list), 2) if counts_list else 0
            if stats['min_samples_per_class'] > 0:
                stats['class_balance_ratio'] = round(stats['max_samples_per_class'] / stats['min_samples_per_class'], 2)
            else:
                stats['class_balance_ratio'] = 0
        
    except Exception as e:
        stats['error'] = str(e)
        stats['exists'] = False
    
    return stats

def print_dataset_statistics(stats):
    """Print formatted statistics for a dataset"""
    
    dataset_name = stats['dataset_name'].upper()
    
    print(f"\n{'=' * 60}")
    print(f"📊 {dataset_name} DATASET STATISTICS")
    print(f"{'=' * 60}")
    
    if not stats.get('exists', False):
        print(f"❌ {dataset_name} Dataset Not Found")
        print(f"Error: {stats.get('error', 'Unknown error')}")
        return
    
    # Print overview
    print(f"Dataset Path: {stats.get('dataset_path', stats.get('source', 'N/A'))}")
    print(f"Total Classes: {stats.get('num_classes', 0)}")
    print(f"Total Samples: {stats.get('total_samples', 0):,}")
    print(f"Avg Samples/Class: {stats.get('avg_samples_per_class', 0)}")
    print(f"Class Balance Ratio: {stats.get('class_balance_ratio', 0)} (max/min)")
    
    # Print split information if available
    if 'split_counts' in stats:
        print(f"\nSplit Distribution:")
        for split_name, count in stats['split_counts'].items():
            percentage = (count / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
            print(f"  {split_name.capitalize()}: {count:,} samples ({percentage:.1f}%)")
    
    # Print detailed class distribution
    if stats.get('class_counts'):
        print(f"\n{'-' * 40}")
        print(f"{dataset_name} CLASS DISTRIBUTION")
        print(f"{'-' * 40}")
        print(f"{'Class Name':<40} {'Count':<10} {'Percentage'}")
        print(f"{'-' * 60}")
        
        total_samples = stats['total_samples']
        
        # Sort classes by sample count (descending)
        sorted_classes = sorted(
            stats['class_counts'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for class_name, count in sorted_classes:
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            # Truncate long class names for display
            display_name = class_name[:35] + "..." if len(class_name) > 35 else class_name
            print(f"{display_name:<40} {count:<10,} {percentage:>6.1f}%")
    
    # Print class names list
    if stats.get('class_names'):
        print(f"\n{'-' * 40}")
        print(f"CLASS NAMES ({len(stats['class_names'])} total):")
        print(f"{'-' * 40}")
        for i, class_name in enumerate(stats['class_names'], 1):
            print(f"  {i:2d}. {class_name}")

def save_statistics_report(all_stats, save_path):
    """Save detailed statistics report to JSON"""
    
    report = {
        'timestamp': str(Path().cwd()),
        'datasets': all_stats,
        'summary': {
            'total_datasets': len([s for s in all_stats if s.get('exists', False)]),
            'total_samples_across_all': sum(s.get('total_samples', 0) for s in all_stats if s.get('exists', False)),
            'unique_classes_combined': len(set().union(*[
                s.get('class_names', []) for s in all_stats if s.get('exists', False)
            ]))
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"💾 Statistics report saved to: {save_path}")

def main():
    """Main function to analyze all datasets"""
    
    print("🔍 DATASET STATISTICS ANALYSIS")
    print("Analyzing plantVillage, tomatoVillage, and combined datasets...")
    
    all_stats = []
    
    # Analyze plantVillage dataset
    print(f"\nAnalyzing plantVillage dataset...")
    plantvillage_config = get_dataset_config('plantvillage')
    plantvillage_stats = get_dataset_statistics(plantvillage_config['root'], 'plantVillage')
    all_stats.append(plantvillage_stats)
    print_dataset_statistics(plantvillage_stats)
    
    # Analyze tomatoVillage dataset
    print(f"\nAnalyzing tomatoVillage dataset...")
    tomatovillage_config = get_dataset_config('tomatovillage')
    tomatovillage_stats = get_split_dataset_statistics(
        tomatovillage_config['train_dir'],
        tomatovillage_config['val_dir'], 
        tomatovillage_config['test_dir'],
        'tomatoVillage'
    )
    all_stats.append(tomatovillage_stats)
    print_dataset_statistics(tomatovillage_stats)
    
    # Analyze combined dataset
    print(f"\nAnalyzing combined dataset...")
    combined_stats = get_combined_dataset_statistics()
    all_stats.append(combined_stats)
    print_dataset_statistics(combined_stats)
    
    # Create summary comparison
    print(f"\n{'=' * 80}")
    print("📋 DATASETS COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<15} {'Status':<8} {'Classes':<8} {'Total Samples':<15} {'Avg/Class':<10} {'Balance'}")
    print(f"{'-' * 80}")
    
    for stats in all_stats:
        status = "✅" if stats.get('exists', False) else "❌"
        classes = str(stats.get('num_classes', 0))
        total = f"{stats.get('total_samples', 0):,}"
        avg = f"{stats.get('avg_samples_per_class', 0):.1f}"
        balance = f"{stats.get('class_balance_ratio', 0):.1f}"
        
        print(f"{stats['dataset_name']:<15} {status:<8} {classes:<8} {total:<15} {avg:<10} {balance}")
    
    # Save outputs
    output_dir = Path("outputs") / "dataset_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_dir / "dataset_statistics.json"
    save_statistics_report(all_stats, json_path)
    
    # Print final summary
    print(f"\n✅ Dataset statistics analysis completed!")
    print(f"Results saved to: {output_dir}")
    
    total_datasets = len([s for s in all_stats if s.get('exists', False)])
    total_samples = sum(s.get('total_samples', 0) for s in all_stats if s.get('exists', False))
    
    print(f"\n📊 Overall Summary:")
    print(f"• {total_datasets} datasets analyzed")
    print(f"• {total_samples:,} total samples across all datasets")
    print(f"• Detailed statistics saved to JSON")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 