#!/usr/bin/env python3
"""
Systematic Data Balancing Experiments for Research Paper
Comprehensive evaluation of data balancing techniques
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID

from data_balancing_config import DataBalancingExperiments

console = Console()

def run_single_experiment(exp_name: str, exp_config: dict, base_cmd: list) -> bool:
    """Run a single data balancing experiment"""
    console.print(f"\n[bold blue]Running: {exp_config['name']}[/bold blue]")
    console.print(f"[dim]Description: {exp_config['description']}[/dim]")
    
    # Prepare command
    cmd = base_cmd + [
        '--experiment_name', f"balance_{exp_name}"
    ]
    
    # Add balancing technique if specified
    if exp_config['technique']:
        cmd.extend(['--balancing_technique', exp_config['technique']])
    
    # Add loss function
    cmd.extend(['--loss_function', exp_config['loss_function']])
    
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        console.print(f"[green]✓ Completed in {elapsed/60:.1f} minutes[/green]")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        console.print(f"[red]✗ Failed with return code: {e.returncode} after {elapsed/60:.1f} minutes[/red]")
        return False

def run_experiment_group(group_name: str, base_cmd: list) -> dict:
    """Run all experiments in a specific group"""
    console.print(f"\n[bold green]Starting Group: {group_name.replace('_', ' ').title()}[/bold green]")
    
    experiments = DataBalancingExperiments.get_experiments_by_group(group_name)
    total_experiments = len(experiments)
    
    if total_experiments == 0:
        console.print(f"[yellow]No experiments found for group: {group_name}[/yellow]")
        return {}
    
    console.print(f"Total experiments in this group: {total_experiments}")
    
    # Create experiment summary table
    table = Table(title=f"{group_name.replace('_', ' ').title()} Experiments")
    table.add_column("Experiment", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Technique", style="yellow")
    table.add_column("Loss Function", style="magenta")
    
    for exp_name, exp_config in experiments.items():
        technique = exp_config.get('technique', 'None')
        loss_fn = exp_config.get('loss_function', 'cross_entropy')
        table.add_row(exp_name, exp_config['name'], technique, loss_fn)
    
    console.print(table)
    
    # Run experiments
    results = {}
    failed_experiments = []
    
    for i, (exp_name, exp_config) in enumerate(experiments.items(), 1):
        console.print(f"\n[cyan]Experiment {i}/{total_experiments} in {group_name}[/cyan]")
        
        success = run_single_experiment(exp_name, exp_config, base_cmd)
        results[exp_name] = 'success' if success else 'failed'
        
        if not success:
            failed_experiments.append((exp_name, exp_config['name']))
        
        # Brief pause between experiments
        if i < total_experiments:
            time.sleep(2)
    
    # Group summary
    successful = sum(1 for r in results.values() if r == 'success')
    console.print(f"\n[bold]Group {group_name} Summary:[/bold]")
    console.print(f"Successful: {successful}/{total_experiments}")
    console.print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        console.print("[red]Failed experiments:[/red]")
        for exp_name, exp_display_name in failed_experiments:
            console.print(f"  - {exp_name}: {exp_display_name}")
    
    return results

def create_results_summary(all_results: dict, save_dir: Path):
    """Create a comprehensive results summary for the research paper"""
    summary_path = save_dir / "balancing_experiments_summary.json"
    
    # Organize results by group
    summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "experiment_type": "data_balancing",
        "total_experiments": sum(len(group_results) for group_results in all_results.values()),
        "groups": {}
    }
    
    for group_name, group_results in all_results.items():
        successful = sum(1 for r in group_results.values() if r == 'success')
        total = len(group_results)
        
        summary["groups"][group_name] = {
            "total_experiments": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "experiments": group_results
        }
    
    # Save summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"\n[bold]Results summary saved to:[/bold] {summary_path}")
    return summary

def print_final_summary(all_results: dict):
    """Print comprehensive final summary"""
    console.print("\n[bold magenta]═══ FINAL DATA BALANCING EXPERIMENTS SUMMARY ═══[/bold magenta]")
    
    total_experiments = sum(len(group_results) for group_results in all_results.values())
    total_successful = sum(
        sum(1 for r in group_results.values() if r == 'success') 
        for group_results in all_results.values()
    )
    total_failed = total_experiments - total_successful
    
    # Overall summary table
    overall_table = Table(title="Overall Results")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="green")
    
    overall_table.add_row("Total Experiments", str(total_experiments))
    overall_table.add_row("Successful", str(total_successful))
    overall_table.add_row("Failed", str(total_failed))
    overall_table.add_row("Success Rate", f"{total_successful/total_experiments*100:.1f}%")
    
    console.print(overall_table)
    
    # Group-wise summary table
    group_table = Table(title="Results by Group")
    group_table.add_column("Group", style="cyan")
    group_table.add_column("Total", style="blue")
    group_table.add_column("Success", style="green")
    group_table.add_column("Failed", style="red")
    group_table.add_column("Rate", style="yellow")
    
    for group_name, group_results in all_results.items():
        successful = sum(1 for r in group_results.values() if r == 'success')
        total = len(group_results)
        failed = total - successful
        rate = successful / total * 100 if total > 0 else 0
        
        group_table.add_row(
            group_name.replace('_', ' ').title(),
            str(total),
            str(successful),
            str(failed),
            f"{rate:.1f}%"
        )
    
    console.print(group_table)

def print_technique_overview():
    """Print overview of data balancing techniques"""
    console.print("\n[bold cyan]Data Balancing Techniques Overview:[/bold cyan]")
    
    techniques_table = Table(title="Balancing Techniques")
    techniques_table.add_column("Technique", style="yellow")
    techniques_table.add_column("Type", style="green")
    techniques_table.add_column("Description", style="white")
    
    techniques_table.add_row(
        "Random Oversampling", 
        "Resampling", 
        "Randomly duplicate minority class samples"
    )
    techniques_table.add_row(
        "SMOTE", 
        "Resampling", 
        "Synthetic Minority Oversampling Technique"
    )
    techniques_table.add_row(
        "ADASYN", 
        "Resampling", 
        "Adaptive Synthetic Sampling"
    )
    techniques_table.add_row(
        "Offline Augmentation", 
        "Augmentation", 
        "Pre-generate augmented samples for balancing"
    )
    techniques_table.add_row(
        "Focal Loss", 
        "Loss Function", 
        "Loss function focusing on hard examples"
    )
    
    console.print(techniques_table)

def main():
    parser = argparse.ArgumentParser(description='Run Systematic Data Balancing Experiments')
    parser.add_argument('--group', type=str, choices=DataBalancingExperiments.get_experiment_groups(),
                        help='Run experiments for specific group only')
    parser.add_argument('--model', type=str, default='shufflenet_v2',
                        help='Model architecture to test (default: shufflenet_v2)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['plantvillage', 'tomatovillage', 'combined'],
                        help='Dataset to use (default: combined)')
    
    args = parser.parse_args()
    
    console.print("[bold magenta]Systematic Data Balancing Experiments[/bold magenta]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print technique overview
    print_technique_overview()
    
    # Experiment configuration
    console.print("\n[bold]Experiment Configuration:[/bold]")
    console.print(f"• Model: {args.model}")
    console.print(f"• Dataset: {args.dataset}")
    console.print(f"• Epochs: {args.epochs}")
    console.print("• Objective: Systematic data balancing evaluation")
    console.print("• Using best hyperparameters and augmentation configuration")
    
    # Base command for all experiments
    base_cmd = [
        'python', 'train_balancing.py',
        '--model', args.model,
        '--dataset', args.dataset,
        '--epochs', str(args.epochs)
    ]
    
    # Create results directory
    results_dir = Path("outputs") / "balancing_experiments" 
    results_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    all_results = {}
    
    if args.group:
        # Run specific group only
        group_results = run_experiment_group(args.group, base_cmd)
        all_results[args.group] = group_results
    else:
        # Run all groups in order
        experiment_groups = DataBalancingExperiments.get_experiment_groups()
        
        console.print(f"\n[bold]Experiment Groups to Run ({len(experiment_groups)}):[/bold]")
        for i, group in enumerate(experiment_groups, 1):
            console.print(f"{i}. {group.replace('_', ' ').title()}")
        
        for group_name in experiment_groups:
            group_results = run_experiment_group(group_name, base_cmd)
            all_results[group_name] = group_results
            
            # Save intermediate results
            create_results_summary(all_results, results_dir)
    
    # Final summary
    total_time = time.time() - start_time
    console.print(f"\n[bold]Total Runtime: {total_time/3600:.1f} hours[/bold]")
    console.print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create final results summary
    summary = create_results_summary(all_results, results_dir)
    print_final_summary(all_results)
    
    # Create experiment recommendations
    console.print("\n[bold yellow]Experiment Analysis Recommendations:[/bold yellow]")
    console.print("• Compare baseline vs. each balancing technique")
    console.print("• Analyze per-class performance improvements")
    console.print("• Evaluate computational overhead of each technique") 
    console.print("• Focus on macro F1-score for imbalanced evaluation")
    console.print("• Consider combining best techniques for further improvement")
    
    # Success/failure summary
    total_experiments = summary["total_experiments"]
    total_successful = sum(group["successful"] for group in summary["groups"].values())
    
    if total_successful == total_experiments:
        console.print("\n[green]🎉 All experiments completed successfully![/green]")
        console.print("Ready for comprehensive data balancing analysis!")
    else:
        console.print(f"\n[yellow]⚠️  {total_experiments - total_successful} experiments failed.[/yellow]")
        console.print("Check the logs and results summary for details.")
    
    console.print(f"\n[bold]All results saved in:[/bold] {results_dir}")

if __name__ == '__main__':
    main() 