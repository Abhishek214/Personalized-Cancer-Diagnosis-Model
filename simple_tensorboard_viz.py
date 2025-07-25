# Simple TensorBoard Visualization Scripts for Multi-Run Directory Structure

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import re
from datetime import datetime
import subprocess
import webbrowser
import time

def discover_training_runs(main_log_dir):
    """
    Discover all training runs in the main directory
    
    Args:
        main_log_dir (str): Path to main TensorBoard logs directory
        
    Returns:
        list: List of run directories (timestamped folders)
    """
    runs = []
    timestamp_pattern = re.compile(r'\d{8}-\d{6}')
    
    for item in os.listdir(main_log_dir):
        item_path = os.path.join(main_log_dir, item)
        if os.path.isdir(item_path) and timestamp_pattern.match(item):
            runs.append(item)
    
    runs.sort()
    return runs

def launch_tensorboard_multirun(main_log_dir, port=6006):
    """
    Launch TensorBoard for multiple runs
    
    Args:
        main_log_dir (str): Path to main TensorBoard logs directory
        port (int): Port number for TensorBoard server
    """
    try:
        cmd = f"tensorboard --logdir={main_log_dir} --port={port}"
        print(f"Launching TensorBoard with command: {cmd}")
        
        process = subprocess.Popen(cmd.split())
        time.sleep(3)
        
        url = f"http://localhost:{port}"
        print(f"Opening TensorBoard at: {url}")
        webbrowser.open(url)
        
        print("Press Ctrl+C to stop TensorBoard")
        process.wait()
        
    except KeyboardInterrupt:
        print("Stopping TensorBoard...")
        process.terminate()

def quick_compare_all_runs(main_log_dir, metric_filter='loss', save_plot=True):
    """
    Quick comparison of all training runs
    
    Args:
        main_log_dir (str): Path to main TensorBoard logs directory
        metric_filter (str): Filter for metrics containing this string
        save_plot (bool): Whether to save the plot
    """
    
    runs = discover_training_runs(main_log_dir)
    if not runs:
        print("No training runs found!")
        return
    
    print(f"Found {len(runs)} training runs: {runs}")
    
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab10(range(len(runs)))
    
    for idx, run in enumerate(runs):
        run_dir = os.path.join(main_log_dir, run)
        event_files = glob.glob(os.path.join(run_dir, "**/events.out.tfevents.*"), recursive=True)
        
        run_data = {'steps': [], 'values': []}
        
        for event_file in event_files:
            try:
                for event in summary_iterator(event_file):
                    if event.summary:
                        for value in event.summary.value:
                            if (metric_filter.lower() in value.tag.lower() and 
                                value.HasField('simple_value')):
                                run_data['steps'].append(event.step)
                                run_data['values'].append(value.simple_value)
            except Exception as e:
                print(f"Error reading {event_file}: {e}")
                continue
        
        if run_data['steps'] and run_data['values']:
            plt.plot(run_data['steps'], run_data['values'], 
                    label=f'Run {run}', color=colors[idx], linewidth=2)
    
    plt.title(f'{metric_filter.title()} Comparison - All Training Runs', fontsize=16, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel(metric_filter.title(), fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        filename = f'all_runs_{metric_filter}_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
    
    plt.show()

def plot_latest_run_detailed(main_log_dir, save_plot=True):
    """
    Plot detailed metrics for the latest training run
    
    Args:
        main_log_dir (str): Path to main TensorBoard logs directory
        save_plot (bool): Whether to save the plot
    """
    
    runs = discover_training_runs(main_log_dir)
    if not runs:
        print("No training runs found!")
        return
    
    latest_run = runs[-1]
    print(f"Plotting detailed metrics for latest run: {latest_run}")
    
    run_dir = os.path.join(main_log_dir, latest_run)
    event_files = glob.glob(os.path.join(run_dir, "**/events.out.tfevents.*"), recursive=True)
    
    metrics_data = {}
    
    for event_file in event_files:
        # Extract metric name from path
        rel_path = os.path.relpath(event_file, run_dir)
        metric_name = rel_path.split(os.sep)[0] if os.sep in rel_path else 'main'
        
        if metric_name not in metrics_data:
            metrics_data[metric_name] = {'steps': [], 'values': []}
        
        try:
            for event in summary_iterator(event_file):
                if event.summary:
                    for value in event.summary.value:
                        if value.HasField('simple_value'):
                            metrics_data[metric_name]['steps'].append(event.step)
                            metrics_data[metric_name]['values'].append(value.simple_value)
        except Exception as e:
            print(f"Error reading {event_file}: {e}")
            continue
    
    # Filter out empty metrics
    metrics_data = {k: v for k, v in metrics_data.items() if v['steps']}
    
    if not metrics_data:
        print("No metrics data found!")
        return
    
    # Create subplots
    n_metrics = len(metrics_data)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (metric_name, data) in enumerate(metrics_data.items()):
        ax = axes[idx] if n_metrics > 1 else axes[0]
        
        ax.plot(data['steps'], data['values'], linewidth=2, color='blue')
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Detailed Metrics - Run {latest_run}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plot:
        filename = f'detailed_metrics_{latest_run}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
    
    plt.show()

def export_all_runs_summary(main_log_dir, output_file='all_runs_summary.csv'):
    """
    Export summary statistics for all runs
    
    Args:
        main_log_dir (str): Path to main TensorBoard logs directory
        output_file (str): Output CSV filename
    """
    
    runs = discover_training_runs(main_log_dir)
    if not runs:
        print("No training runs found!")
        return
    
    summary_data = []
    
    for run in runs:
        run_dir = os.path.join(main_log_dir, run)
        event_files = glob.glob(os.path.join(run_dir, "**/events.out.tfevents.*"), recursive=True)
        
        run_metrics = {}
        
        for event_file in event_files:
            rel_path = os.path.relpath(event_file, run_dir)
            metric_name = rel_path.split(os.sep)[0] if os.sep in rel_path else 'main'
            
            if metric_name not in run_metrics:
                run_metrics[metric_name] = []
            
            try:
                for event in summary_iterator(event_file):
                    if event.summary:
                        for value in event.summary.value:
                            if value.HasField('simple_value'):
                                run_metrics[metric_name].append({
                                    'step': event.step,
                                    'value': value.simple_value,
                                    'wall_time': event.wall_time
                                })
            except Exception as e:
                print(f"Error reading {event_file}: {e}")
                continue
        
        # Calculate summary stats for each metric
        for metric_name, data_points in run_metrics.items():
            if data_points:
                values = [dp['value'] for dp in data_points]
                steps = [dp['step'] for dp in data_points]
                times = [dp['wall_time'] for dp in data_points]
                
                summary_data.append({
                    'run': run,
                    'metric': metric_name,
                    'final_value': values[-1],
                    'min_value': min(values),
                    'max_value': max(values),
                    'mean_value': sum(values) / len(values),
                    'total_steps': len(values),
                    'final_step': max(steps) if steps else 0,
                    'duration_hours': (max(times) - min(times)) / 3600 if len(times) > 1 else 0
                })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Summary exported to {output_file}")
    return df

def compare_specific_runs(main_log_dir, run_names, metric_filter='loss'):
    """
    Compare specific runs
    
    Args:
        main_log_dir (str): Path to main TensorBoard logs directory
        run_names (list): List of run names to compare
        metric_filter (str): Metric to filter for
    """
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(range(len(run_names)))
    
    for idx, run_name in enumerate(run_names):
        run_dir = os.path.join(main_log_dir, run_name)
        
        if not os.path.exists(run_dir):
            print(f"Run directory not found: {run_dir}")
            continue
        
        event_files = glob.glob(os.path.join(run_dir, "**/events.out.tfevents.*"), recursive=True)
        
        run_data = {'steps': [], 'values': []}
        
        for event_file in event_files:
            try:
                for event in summary_iterator(event_file):
                    if event.summary:
                        for value in event.summary.value:
                            if (metric_filter.lower() in value.tag.lower() and 
                                value.HasField('simple_value')):
                                run_data['steps'].append(event.step)
                                run_data['values'].append(value.simple_value)
            except Exception as e:
                print(f"Error reading {event_file}: {e}")
                continue
        
        if run_data['steps'] and run_data['values']:
            plt.plot(run_data['steps'], run_data['values'], 
                    label=f'Run {run_name}', color=colors[idx], linewidth=2)
    
    plt.title(f'{metric_filter.title()} Comparison - Selected Runs', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step')
    plt.ylabel(metric_filter.title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def efficientdet_quick_analysis(main_log_dir="/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhi1/tensorboard/"):
    """
    Quick analysis function specifically for EfficientDet logs
    
    Args:
        main_log_dir (str): Path to main tensorboard directory
    """
    
    print(f"Quick analysis of EfficientDet logs in: {main_log_dir}")
    
    # Discover runs
    runs = discover_training_runs(main_log_dir)
    print(f"Found {len(runs)} training runs: {runs}")
    
    if not runs:
        print("No training runs found!")
        return
    
    # 1. Compare all runs - Loss
    print("\n1. Comparing Loss across all runs...")
    quick_compare_all_runs(main_log_dir, metric_filter='loss')
    
    # 2. Compare Regression Loss
    print("\n2. Comparing Regression Loss across all runs...")
    quick_compare_all_runs(main_log_dir, metric_filter='regression')
    
    # 3. Compare Classification Loss
    print("\n3. Comparing Classification Loss across all runs...")
    quick_compare_all_runs(main_log_dir, metric_filter='classification')
    
    # 4. Detailed view of latest run
    print("\n4. Detailed metrics for latest run...")
    plot_latest_run_detailed(main_log_dir)
    
    # 5. Export summary
    print("\n5. Exporting summary statistics...")
    summary_df = export_all_runs_summary(main_log_dir)
    if not summary_df.empty:
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))
    
    return runs, summary_df

# Example usage functions
def launch_tensorboard_for_efficientdet():
    """Launch TensorBoard for EfficientDet logs"""
    main_log_dir = "/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhi1/tensorboard/"
    launch_tensorboard_multirun(main_log_dir)

def jupyter_tensorboard_efficientdet():
    """Generate Jupyter notebook code for EfficientDet logs"""
    main_log_dir = "/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhi1/tensorboard/"
    code = f"""
# Load TensorBoard extension
%load_ext tensorboard

# Launch TensorBoard for all runs
%tensorboard --logdir {main_log_dir}
"""
    print("Copy and paste this code into a Jupyter notebook cell:")
    print(code)

if __name__ == "__main__":
    # Quick analysis of your EfficientDet logs
    runs, summary = efficientdet_quick_analysis()
    
    # Optional: Launch TensorBoard (uncomment to use)
    # print("\nLaunching TensorBoard...")
    # launch_tensorboard_for_efficientdet()
    
    # Optional: Show Jupyter code
    print("\nJupyter notebook code:")
    jupyter_tensorboard_efficientdet()
