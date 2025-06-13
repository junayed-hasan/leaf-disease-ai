#!/usr/bin/env python3
"""
Comprehensive ONNX Quantized Model Evaluation
Runs all ONNX quantized model evaluations including combined and individual test sets
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
import time
import traceback

console = Console()

class ONNXEvaluationRunner:
    """Comprehensive ONNX quantized model evaluation runner"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.start_time = datetime.now()
        
        # Create master output directory
        timestamp = self.start_time.strftime('%Y%m%d%H%M%S')
        self.master_dir = Path("outputs") / "onnx_quantized_evaluation" / f"onnx_evaluation_{timestamp}"
        self.master_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold]ONNX Quantized Model Comprehensive Evaluation[/bold]")
        console.print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Master output directory: {self.master_dir}")
    
    def check_onnx_model_exists(self):
        """Check if ONNX quantized model exists"""
        onnx_path = Path("mobile_models/shufflenet_v2_mobile_quantized.onnx")
        
        if onnx_path.exists():
            model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            console.print(f"✓ Found ONNX quantized model: {onnx_path}")
            console.print(f"Model size: {model_size_mb:.2f} MB")
            return True, onnx_path, model_size_mb
        else:
            console.print(f"[red]✗ ONNX quantized model not found at: {onnx_path}[/red]")
            console.print("[yellow]Please ensure you have run the mobile quantization pipeline first[/yellow]")
            return False, None, 0
    
    def run_evaluation_script(self, script_name, description):
        """Run an evaluation script and capture results"""
        console.print(f"\n[bold]{description}[/bold]")
        console.print(f"Running: {script_name}")
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                console.print(f"[green]✓ {description} completed successfully in {duration:.1f}s[/green]")
                self.results[script_name] = {
                    "status": "success",
                    "duration": duration,
                    "description": description,
                    "stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
                    "stderr": result.stderr[-500:] if result.stderr else ""    # Last 500 chars
                }
                return True
            else:
                console.print(f"[red]✗ {description} failed with return code {result.returncode}[/red]")
                self.results[script_name] = {
                    "status": "failed",
                    "duration": duration,
                    "description": description,
                    "return_code": result.returncode,
                    "stdout": result.stdout[-1000:] if result.stdout else "",
                    "stderr": result.stderr[-1000:] if result.stderr else ""
                }
                self.errors.append({
                    "script": script_name,
                    "description": description,
                    "error": result.stderr[-500:] if result.stderr else "Unknown error"
                })
                return False
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            console.print(f"[red]✗ {description} failed with exception: {e}[/red]")
            self.results[script_name] = {
                "status": "exception",
                "duration": duration,
                "description": description,
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
            self.errors.append({
                "script": script_name,
                "description": description,
                "error": str(e)
            })
            return False
    
    def collect_results(self):
        """Collect and summarize all evaluation results"""
        console.print(f"\n[bold]Collecting ONNX Quantized Model Evaluation Results...[/bold]")
        
        collected_results = {
            "combined_test": {},
            "individual_tests": {},
            "summary": {}
        }
        
        # Look for combined test results
        combined_dirs = list(Path("outputs/test_evaluation/onnx_quantized_combined").glob("*"))
        if combined_dirs:
            latest_combined = max(combined_dirs, key=lambda x: x.stat().st_mtime)
            combined_summary_file = latest_combined / "evaluation_summary.json"
            
            if combined_summary_file.exists():
                with open(combined_summary_file, 'r') as f:
                    collected_results["combined_test"] = json.load(f)
                console.print(f"✓ Collected combined test results from {latest_combined}")
        
        # Look for individual test results
        individual_dirs = list(Path("outputs/test_evaluation/onnx_quantized_individual").glob("*"))
        if individual_dirs:
            latest_individual = max(individual_dirs, key=lambda x: x.stat().st_mtime)
            individual_summary_file = latest_individual / "evaluation_summary.json"
            
            if individual_summary_file.exists():
                with open(individual_summary_file, 'r') as f:
                    collected_results["individual_tests"] = json.load(f)
                console.print(f"✓ Collected individual test results from {latest_individual}")
        
        # Create summary
        summary = {
            "evaluation_type": "onnx_quantized_comprehensive",
            "total_evaluations": len(self.results),
            "successful_evaluations": sum(1 for r in self.results.values() if r["status"] == "success"),
            "failed_evaluations": sum(1 for r in self.results.values() if r["status"] != "success"),
            "total_duration": (datetime.now() - self.start_time).total_seconds(),
            "evaluation_scripts": list(self.results.keys()),
            "errors": self.errors
        }
        
        # Add performance summary if available
        if collected_results["combined_test"] and "performance" in collected_results["combined_test"]:
            combined_perf = collected_results["combined_test"]["performance"]
            summary["combined_performance"] = {
                "accuracy": combined_perf.get("accuracy", 0),
                "f1_macro": combined_perf.get("f1_macro", 0),
                "f1_weighted": combined_perf.get("f1_weighted", 0),
                "total_samples": combined_perf.get("total_samples", 0),
                "inference_time": combined_perf.get("avg_inference_time_per_batch", 0),
                "model_size_mb": combined_perf.get("model_size_mb", 0)
            }
        
        if collected_results["individual_tests"] and "results_by_dataset" in collected_results["individual_tests"]:
            individual_results = collected_results["individual_tests"]["results_by_dataset"]
            summary["individual_performance"] = {}
            for dataset_name, perf in individual_results.items():
                summary["individual_performance"][dataset_name] = {
                    "accuracy": perf.get("accuracy", 0),
                    "f1_macro": perf.get("f1_macro", 0),
                    "f1_weighted": perf.get("f1_weighted", 0),
                    "total_samples": perf.get("total_samples", 0),
                    "inference_time": perf.get("avg_inference_time_per_batch", 0)
                }
        
        collected_results["summary"] = summary
        
        return collected_results
    
    def save_comprehensive_results(self, collected_results):
        """Save comprehensive evaluation results"""
        
        # Save main results
        with open(self.master_dir / "comprehensive_onnx_evaluation_results.json", "w") as f:
            json.dump(collected_results, f, indent=4)
        
        # Save execution results
        execution_results = {
            "execution_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": (datetime.now() - self.start_time).total_seconds(),
                "scripts_executed": len(self.results),
                "successful_scripts": sum(1 for r in self.results.values() if r["status"] == "success"),
                "failed_scripts": sum(1 for r in self.results.values() if r["status"] != "success")
            },
            "script_results": self.results,
            "errors": self.errors
        }
        
        with open(self.master_dir / "execution_results.json", "w") as f:
            json.dump(execution_results, f, indent=4)
        
        console.print(f"✓ Comprehensive results saved to {self.master_dir}")
    
    def print_final_summary(self, collected_results):
        """Print final evaluation summary"""
        
        console.print(f"\n[bold]ONNX Quantized Model Comprehensive Evaluation Summary[/bold]")
        console.print(f"{'='*70}")
        
        # Execution summary
        total_duration = (datetime.now() - self.start_time).total_seconds()
        successful = sum(1 for r in self.results.values() if r["status"] == "success")
        total_scripts = len(self.results)
        
        console.print(f"Execution Summary:")
        console.print(f"  Total Scripts: {total_scripts}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Failed: {total_scripts - successful}")
        console.print(f"  Total Duration: {total_duration:.1f}s")
        
        # Performance summary
        if "combined_performance" in collected_results["summary"]:
            perf = collected_results["summary"]["combined_performance"]
            console.print(f"\nCombined Test Performance:")
            console.print(f"  Accuracy: {perf['accuracy']:.2f}%")
            console.print(f"  F1 Score (Macro): {perf['f1_macro']:.2f}%")
            console.print(f"  F1 Score (Weighted): {perf['f1_weighted']:.2f}%")
            console.print(f"  Test Samples: {perf['total_samples']}")
            console.print(f"  Inference Time: {perf['inference_time']:.4f}s per batch")
            console.print(f"  Model Size: {perf['model_size_mb']:.2f} MB")
        
        if "individual_performance" in collected_results["summary"]:
            console.print(f"\nIndividual Test Performance:")
            for dataset, perf in collected_results["summary"]["individual_performance"].items():
                console.print(f"  {dataset.upper()}:")
                console.print(f"    Accuracy: {perf['accuracy']:.2f}%")
                console.print(f"    F1 Score (Macro): {perf['f1_macro']:.2f}%")
                console.print(f"    Test Samples: {perf['total_samples']}")
                console.print(f"    Inference Time: {perf['inference_time']:.4f}s per batch")
        
        # Error summary
        if self.errors:
            console.print(f"\n[yellow]Errors Encountered:[/yellow]")
            for error in self.errors:
                console.print(f"  {error['script']}: {error['error'][:100]}...")
        
        console.print(f"\n[green]ONNX quantized model comprehensive evaluation completed![/green]")
        console.print(f"Results saved to: {self.master_dir}")
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive ONNX quantized model evaluation"""
        
        # Check if ONNX model exists
        model_exists, model_path, model_size = self.check_onnx_model_exists()
        if not model_exists:
            console.print("[red]Cannot proceed without ONNX quantized model[/red]")
            return False
        
        # List of evaluation scripts to run
        evaluations = [
            {
                "script": "evaluate_onnx_quantized_on_combined_test.py",
                "description": "ONNX Quantized Model on Combined Test Dataset"
            },
            {
                "script": "evaluate_onnx_quantized_on_test_datasets.py", 
                "description": "ONNX Quantized Model on Individual Test Datasets"
            }
        ]
        
        # Run each evaluation
        for eval_config in evaluations:
            success = self.run_evaluation_script(
                eval_config["script"], 
                eval_config["description"]
            )
            
            if not success:
                console.print(f"[yellow]Continuing with next evaluation despite failure...[/yellow]")
        
        # Collect all results
        collected_results = self.collect_results()
        
        # Save comprehensive results
        self.save_comprehensive_results(collected_results)
        
        # Print final summary
        self.print_final_summary(collected_results)
        
        # Return success if at least one evaluation succeeded
        successful_evaluations = sum(1 for r in self.results.values() if r["status"] == "success")
        return successful_evaluations > 0

def main():
    """Main function to run comprehensive ONNX quantized model evaluation"""
    
    try:
        runner = ONNXEvaluationRunner()
        success = runner.run_comprehensive_evaluation()
        return success
    
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Evaluation interrupted by user[/yellow]")
        return False
    
    except Exception as e:
        console.print(f"\n[red]Evaluation failed with exception: {e}[/red]")
        console.print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 