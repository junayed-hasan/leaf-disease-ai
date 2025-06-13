#!/usr/bin/env python3
"""
Knowledge Distillation Experiments Runner
Systematic experimentation with different KD configurations
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

def run_experiment(cmd, exp_num, total_exps, description):
    """Run a single experiment"""
    console.print(f"\n[bold blue]Experiment {exp_num}/{total_exps}: {description}[/bold blue]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        console.print(f"[green]✓ Completed in {elapsed:.1f}s[/green]")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        console.print(f"[red]✗ Failed with return code: {e.returncode}[/red]")
        console.print(f"[red]Failed after {elapsed:.1f}s[/red]")
        return False

def get_alpha_beta_experiments():
    """Generate alpha/beta combinations where alpha + beta = 1.0"""
    experiments = []
    
    # Alpha values from 0.0 to 1.0 in increments of 0.1
    alpha_values = [round(i * 0.1, 1) for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    
    for alpha in alpha_values:
        beta = round(1.0 - alpha, 1)  # Ensure beta = 1.0 - alpha
        experiments.append({
            'alpha': alpha,
            'beta': beta,
            'gamma': 0.0,  # No feature distillation
            'temperature': 4.0,  # Standard temperature
            'description': f'Basic KD (α={alpha}, β={beta})'
        })
    
    return experiments

def run_step_1():
    """Step 1: Basic Knowledge Distillation with different alpha/beta combinations"""
    console.print("\n[bold green]Step 1: Basic Knowledge Distillation - Alpha/Beta Exploration[/bold green]")
    console.print("Testing different combinations of classification loss (α) and distillation loss (β)")
    console.print("Where α + β = 1.0, α ∈ [0.0, 0.1, 0.2, ..., 1.0]")
    
    experiments = get_alpha_beta_experiments()
    total_experiments = len(experiments)
    
    console.print(f"\nTotal experiments: {total_experiments}")
    
    # Create summary table
    table = Table(title="Experiment Plan")
    table.add_column("Exp", style="cyan", no_wrap=True)
    table.add_column("Alpha (α)", style="green")
    table.add_column("Beta (β)", style="blue") 
    table.add_column("Temperature", style="yellow")
    table.add_column("Description", style="white")
    
    for i, exp in enumerate(experiments, 1):
        table.add_row(
            str(i),
            str(exp['alpha']),
            str(exp['beta']),
            str(exp['temperature']),
            exp['description']
        )
    
    console.print(table)
    
    failed_experiments = []
    
    for i, exp in enumerate(experiments, 1):
        cmd = [
            'python', 'train_kd.py',
            '--student_model', 'shufflenet_v2',
            '--alpha', str(exp['alpha']),
            '--beta', str(exp['beta']),
            '--gamma', str(exp['gamma']),
            '--temperature', str(exp['temperature']),
            '--dataset', 'combined',
            '--epochs', '50',
            '--batch_size', '32',
            '--learning_rate', '0.001'
        ]
        
        success = run_experiment(cmd, i, total_experiments, exp['description'])
        if not success:
            failed_experiments.append((i, exp['description']))
    
    # Print summary
    console.print(f"\n[bold]Step 1 Summary:[/bold]")
    console.print(f"Total experiments: {total_experiments}")
    console.print(f"Successful: {total_experiments - len(failed_experiments)}")
    console.print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        console.print("\n[red]Failed experiments:[/red]")
        for exp_num, description in failed_experiments:
            console.print(f"  {exp_num}: {description}")
    
    return len(failed_experiments) == 0

def main():
    parser = argparse.ArgumentParser(description='Run Knowledge Distillation Experiments')
    parser.add_argument('--step', type=int, choices=[1], default=1,
                        help='Experiment step to run (only step 1 available)')
    
    args = parser.parse_args()
    
    console.print("[bold magenta]Knowledge Distillation Experiments[/bold magenta]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Experiment configuration
    console.print("\n[bold]Experiment Configuration:[/bold]")
    console.print("• Student Model: ShuffleNetV2")
    console.print("• Teacher Ensemble: DenseNet-121 + ResNet-101 + DenseNet-201 + EfficientNet-B4")
    console.print("• Dataset: Combined (15 classes)")
    console.print("• Training: 50 epochs, batch size 32, learning rate 0.001")
    console.print("• Knowledge Distillation: Basic only (no feature distillation)")
    
    start_time = time.time()
    
    if args.step == 1:
        success = run_step_1()
    else:
        console.print(f"[red]Invalid step: {args.step}[/red]")
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    console.print(f"\n[bold]Total Runtime: {total_time/3600:.1f} hours[/bold]")
    console.print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        console.print("[green]All experiments completed successfully![/green]")
    else:
        console.print("[red]Some experiments failed. Check the logs above.[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main() 