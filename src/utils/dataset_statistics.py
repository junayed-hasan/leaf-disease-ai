#!/usr/bin/env python3
"""
Dataset Statistics Analysis
Prints comprehensive statistics for all datasets: plantVillage, tomatoVillage, and combined
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from config import *
from data_loader import load_data_single_directory, load_data_split_directory
from test_dataset_loader import get_available_test_datasets, load_original_class_mapping

console = Console()

def get_dataset_statistics(dataset_path: Path, dataset_name: str) -> dict:
    """Get comprehensive statistics for a dataset"""
    
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

def get_combined_dataset_statistics() -> dict:
    """Get statistics for the combined dataset using the data loader"""
    
    stats = {
        'dataset_name': 'combined',
        'exists': True,
        'source': 'config.py data paths'
    }
    
    try:
        # Use the original data loading approach
        combined_data_paths = [TOMATOVILLAGE_DATA_PATH, PLANTVILLAGE_DATA_PATH]
        
        # Load combined data to get class mapping
        class_mapping = load_original_class_mapping()
        
        stats['class_names'] = sorted(class_mapping.keys())
        stats['num_classes'] = len(class_mapping)
        
        # Count samples across all source datasets
        total_samples = 0
        class_counts = defaultdict(int)
        
        for data_path in combined_data_paths:
            dataset_path = Path(data_path)
            if dataset_path.exists():
                for class_dir in dataset_path.iterdir():
                    if class_dir.is_dir() and not class_dir.name.startswith('.'):
                        class_name = class_dir.name
                        
                        # Map to combined class name
                        combined_class = None
                        for combined_name, idx in class_mapping.items():
                            if class_name.lower() in combined_name.lower() or combined_name.lower() in class_name.lower():
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
            stats['min_samples_per_class'] = min(counts_list)
            stats['max_samples_per_class'] = max(counts_list)
            stats['avg_samples_per_class'] = round(sum(counts_list) / len(counts_list), 2)
            stats['class_balance_ratio'] = round(stats['max_samples_per_class'] / stats['min_samples_per_class'], 2)
        
    except Exception as e:
        stats['error'] = str(e)
    
    return stats

def print_dataset_statistics(stats: dict):
    """Print formatted statistics for a dataset"""
    
    dataset_name = stats['dataset_name'].upper()
    
    if not stats.get('exists', False):
        console.print(f"\n[red]❌ {dataset_name} Dataset Not Found[/red]")
        console.print(f"[red]Error: {stats.get('error', 'Unknown error')}[/red]")
        return
    
    # Create main info panel
    info_text = f"""
[bold]Dataset Path:[/bold] {stats.get('dataset_path', stats.get('source', 'N/A'))}
[bold]Total Classes:[/bold] {stats.get('num_classes', 0)}
[bold]Total Samples:[/bold] {stats.get('total_samples', 0):,}
[bold]Avg Samples/Class:[/bold] {stats.get('avg_samples_per_class', 0)}
[bold]Class Balance Ratio:[/bold] {stats.get('class_balance_ratio', 0)} (max/min)
    """.strip()
    
    console.print(f"\n[bold cyan]📊 {dataset_name} DATASET STATISTICS[/bold cyan]")
    console.print(Panel(info_text, title=f"{dataset_name} Overview", border_style="cyan"))
    
    # Create detailed class table
    if stats.get('class_counts'):
        table = Table(title=f"{dataset_name} Class Distribution")
        table.add_column("Class Name", style="bold")
        table.add_column("Sample Count", justify="right", style="cyan")
        table.add_column("Percentage", justify="right", style="green")
        
        total_samples = stats['total_samples']
        
        # Sort classes by sample count (descending)
        sorted_classes = sorted(
            stats['class_counts'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for class_name, count in sorted_classes:
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            table.add_row(
                class_name,
                f"{count:,}",
                f"{percentage:.1f}%"
            )
        
        console.print(table)
    
    # Print class names list
    if stats.get('class_names'):
        console.print(f"\n[bold]Class Names ({len(stats['class_names'])}):[/bold]")
        for i, class_name in enumerate(stats['class_names'], 1):
            console.print(f"  {i:2d}. {class_name}")

def create_comparison_visualization(all_stats: list, save_path: Path = None):
    """Create comparison visualizations for all datasets"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Statistics Comparison', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    dataset_names = []
    total_samples = []
    num_classes = []
    avg_samples_per_class = []
    
    for stats in all_stats:
        if stats.get('exists', False) and not stats.get('error'):
            dataset_names.append(stats['dataset_name'])
            total_samples.append(stats.get('total_samples', 0))
            num_classes.append(stats.get('num_classes', 0))
            avg_samples_per_class.append(stats.get('avg_samples_per_class', 0))
    
    # Plot 1: Total Samples
    axes[0, 0].bar(dataset_names, total_samples, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Total Samples per Dataset')
    axes[0, 0].set_ylabel('Number of Samples')
    for i, v in enumerate(total_samples):
        axes[0, 0].text(i, v + max(total_samples) * 0.01, f'{v:,}', ha='center', va='bottom')
    
    # Plot 2: Number of Classes
    axes[0, 1].bar(dataset_names, num_classes, alpha=0.7, color=['orange', 'purple', 'gold'])
    axes[0, 1].set_title('Number of Classes per Dataset')
    axes[0, 1].set_ylabel('Number of Classes')
    for i, v in enumerate(num_classes):
        axes[0, 1].text(i, v + max(num_classes) * 0.01, str(v), ha='center', va='bottom')
    
    # Plot 3: Average Samples per Class
    axes[1, 0].bar(dataset_names, avg_samples_per_class, alpha=0.7, color=['cyan', 'magenta', 'yellow'])
    axes[1, 0].set_title('Average Samples per Class')
    axes[1, 0].set_ylabel('Avg Samples per Class')
    for i, v in enumerate(avg_samples_per_class):
        axes[1, 0].text(i, v + max(avg_samples_per_class) * 0.01, f'{v:.1f}', ha='center', va='bottom')
    
    # Plot 4: Class Distribution for Combined Dataset
    combined_stats = next((s for s in all_stats if s['dataset_name'] == 'combined'), None)
    if combined_stats and combined_stats.get('class_counts'):
        class_names = list(combined_stats['class_counts'].keys())
        class_counts = list(combined_stats['class_counts'].values())
        
        # Truncate long class names for display
        display_names = [name[:20] + '...' if len(name) > 20 else name for name in class_names]
        
        axes[1, 1].barh(range(len(class_names)), class_counts, alpha=0.7)
        axes[1, 1].set_yticks(range(len(class_names)))
        axes[1, 1].set_yticklabels(display_names, fontsize=8)
        axes[1, 1].set_title('Combined Dataset Class Distribution')
        axes[1, 1].set_xlabel('Number of Samples')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]📊 Visualization saved to: {save_path}[/green]")
    
    plt.close()

def save_statistics_report(all_stats: list, save_path: Path):
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
    
    console.print(f"[green]💾 Statistics report saved to: {save_path}[/green]")

def main():
    """Main function to analyze all datasets"""
    
    console.print("[bold cyan]🔍 DATASET STATISTICS ANALYSIS[/bold cyan]")
    console.print("Analyzing plantVillage, tomatoVillage, and combined datasets...\n")
    
    # Define dataset paths
    datasets = [
        {
            'name': 'plantVillage',
            'path': Path(PLANTVILLAGE_DATA_PATH)
        },
        {
            'name': 'tomatoVillage', 
            'path': Path(TOMATOVILLAGE_DATA_PATH)
        }
    ]
    
    all_stats = []
    
    # Analyze individual datasets
    for dataset in datasets:
        console.print(f"[yellow]Analyzing {dataset['name']} dataset...[/yellow]")
        stats = get_dataset_statistics(dataset['path'], dataset['name'])
        all_stats.append(stats)
        print_dataset_statistics(stats)
    
    # Analyze combined dataset
    console.print(f"[yellow]Analyzing combined dataset...[/yellow]")
    combined_stats = get_combined_dataset_statistics()
    all_stats.append(combined_stats)
    print_dataset_statistics(combined_stats)
    
    # Create summary comparison
    console.print("\n[bold magenta]📋 DATASETS COMPARISON SUMMARY[/bold magenta]")
    
    summary_table = Table(title="Dataset Comparison Summary")
    summary_table.add_column("Dataset", style="bold")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Classes", justify="right")
    summary_table.add_column("Total Samples", justify="right")
    summary_table.add_column("Avg/Class", justify="right")
    summary_table.add_column("Balance Ratio", justify="right")
    
    for stats in all_stats:
        status = "✅" if stats.get('exists', False) else "❌"
        classes = str(stats.get('num_classes', 0))
        total = f"{stats.get('total_samples', 0):,}"
        avg = f"{stats.get('avg_samples_per_class', 0):.1f}"
        balance = f"{stats.get('class_balance_ratio', 0):.1f}"
        
        summary_table.add_row(
            stats['dataset_name'],
            status,
            classes,
            total,
            avg,
            balance
        )
    
    console.print(summary_table)
    
    # Save outputs
    output_dir = Path("outputs") / "dataset_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_dir / "dataset_statistics.json"
    save_statistics_report(all_stats, json_path)
    
    # Create and save visualization
    viz_path = output_dir / "dataset_comparison.png"
    create_comparison_visualization(all_stats, viz_path)
    
    # Print final summary
    console.print(f"\n[bold green]✅ Dataset statistics analysis completed![/bold green]")
    console.print(f"[bold]Results saved to:[/bold] {output_dir}")
    
    total_datasets = len([s for s in all_stats if s.get('exists', False)])
    total_samples = sum(s.get('total_samples', 0) for s in all_stats if s.get('exists', False))
    
    console.print(f"\n[bold]📊 Overall Summary:[/bold]")
    console.print(f"• {total_datasets} datasets analyzed")
    console.print(f"• {total_samples:,} total samples across all datasets")
    console.print(f"• Detailed statistics and visualizations saved")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 