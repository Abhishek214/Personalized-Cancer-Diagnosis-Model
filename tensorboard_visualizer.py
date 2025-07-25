import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class TensorBoardVisualizer:
    def __init__(self, log_dir):
        """
        Initialize TensorBoard visualizer
        
        Args:
            log_dir (str): Path to TensorBoard logs directory
        """
        self.log_dir = log_dir
        self.data = defaultdict(list)
        self.scalar_data = {}
        
    def parse_tensorboard_logs(self):
        """Parse TensorBoard event files and extract scalar data"""
        
        # Find all event files recursively
        event_files = glob.glob(os.path.join(self.log_dir, "**/events.out.tfevents.*"), recursive=True)
        
        for event_file in event_files:
            # Get the parent directory name as the run name
            run_name = os.path.basename(os.path.dirname(event_file))
            
            try:
                for event in summary_iterator(event_file):
                    if event.summary:
                        for value in event.summary.value:
                            if value.HasField('simple_value'):
                                metric_name = value.tag
                                step = event.step
                                scalar_value = value.simple_value
                                wall_time = event.wall_time
                                
                                self.data[f"{run_name}/{metric_name}"].append({
                                    'step': step,
                                    'value': scalar_value,
                                    'wall_time': wall_time,
                                    'run': run_name,
                                    'metric': metric_name
                                })
            except Exception as e:
                print(f"Error reading {event_file}: {e}")
                continue
        
        # Convert to DataFrames
        for key, values in self.data.items():
            self.scalar_data[key] = pd.DataFrame(values)
        
        return self.scalar_data
    
    def plot_metrics_matplotlib(self, metrics=None, figsize=(15, 10), style='seaborn-v0_8'):
        """
        Create matplotlib plots for specified metrics
        
        Args:
            metrics (list): List of metric names to plot. If None, plots all metrics
            figsize (tuple): Figure size
            style (str): Matplotlib style
        """
        
        if not self.scalar_data:
            self.parse_tensorboard_logs()
        
        if metrics is None:
            metrics = list(set([key.split('/')[-1] for key in self.scalar_data.keys()]))
        
        plt.style.use(style)
        
        # Calculate subplot layout
        n_metrics = len(metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            color_idx = 0
            for key, df in self.scalar_data.items():
                if metric in key:
                    run_name = key.split('/')[0]
                    ax.plot(df['step'], df['value'], 
                           label=run_name, 
                           color=colors[color_idx % len(colors)],
                           linewidth=2, alpha=0.8)
                    color_idx += 1
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_plotly(self, metrics=None, height=600):
        """
        Create interactive Plotly plots for specified metrics
        
        Args:
            metrics (list): List of metric names to plot. If None, plots all metrics
            height (int): Height of the plot
        """
        
        if not self.scalar_data:
            self.parse_tensorboard_logs()
        
        if metrics is None:
            metrics = list(set([key.split('/')[-1] for key in self.scalar_data.keys()]))
        
        # Create subplots
        n_metrics = len(metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        subplot_titles = [metric.replace('_', ' ').title() for metric in metrics]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for idx, metric in enumerate(metrics):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            color_idx = 0
            for key, df in self.scalar_data.items():
                if metric in key:
                    run_name = key.split('/')[0]
                    fig.add_trace(
                        go.Scatter(
                            x=df['step'],
                            y=df['value'],
                            mode='lines',
                            name=f'{run_name} - {metric}',
                            line=dict(color=colors[color_idx % len(colors)], width=2),
                            hovertemplate=f'<b>{run_name}</b><br>' +
                                        'Step: %{x}<br>' +
                                        'Value: %{y:.4f}<br>' +
                                        '<extra></extra>'
                        ),
                        row=row, col=col
                    )
                    color_idx += 1
        
        fig.update_layout(
            height=height * rows,
            title_text="TensorBoard Metrics Visualization",
            title_x=0.5,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Training Step")
        fig.update_yaxes(title_text="Value")
        
        fig.show()
    
    def compare_runs(self, metric_name, runs=None):
        """
        Compare specific metric across different runs
        
        Args:
            metric_name (str): Name of the metric to compare
            runs (list): List of run names to compare. If None, compares all runs
        """
        
        if not self.scalar_data:
            self.parse_tensorboard_logs()
        
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        color_idx = 0
        
        for key, df in self.scalar_data.items():
            if metric_name in key:
                run_name = key.split('/')[0]
                if runs is None or run_name in runs:
                    plt.plot(df['step'], df['value'], 
                           label=run_name, 
                           color=colors[color_idx % len(colors)],
                           linewidth=2, alpha=0.8)
                    color_idx += 1
        
        plt.title(f'{metric_name.replace("_", " ").title()} Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_summary_stats(self):
        """Get summary statistics for all metrics"""
        
        if not self.scalar_data:
            self.parse_tensorboard_logs()
        
        summary_data = []
        
        for key, df in self.scalar_data.items():
            run_name, metric_name = key.split('/', 1)
            
            stats = {
                'Run': run_name,
                'Metric': metric_name,
                'Final Value': df['value'].iloc[-1] if not df.empty else None,
                'Min Value': df['value'].min() if not df.empty else None,
                'Max Value': df['value'].max() if not df.empty else None,
                'Mean Value': df['value'].mean() if not df.empty else None,
                'Std Value': df['value'].std() if not df.empty else None,
                'Total Steps': len(df) if not df.empty else 0
            }
            summary_data.append(stats)
        
        return pd.DataFrame(summary_data)
    
    def export_data(self, output_path='tensorboard_data.csv'):
        """Export parsed data to CSV"""
        
        if not self.scalar_data:
            self.parse_tensorboard_logs()
        
        all_data = []
        for key, df in self.scalar_data.items():
            df_copy = df.copy()
            run_name, metric_name = key.split('/', 1)
            df_copy['run_metric'] = key
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Data exported to {output_path}")

# Example usage
def main():
    # Replace with your TensorBoard logs directory
    log_dir = "/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhi1/tensorboard/20250725-024509/"
    
    # Initialize visualizer
    viz = TensorBoardVisualizer(log_dir)
    
    # Parse logs
    print("Parsing TensorBoard logs...")
    data = viz.parse_tensorboard_logs()
    print(f"Found {len(data)} metric series")
    
    # Plot all metrics with matplotlib
    print("Creating matplotlib plots...")
    viz.plot_metrics_matplotlib()
    
    # Create interactive plotly plots
    print("Creating interactive plotly plots...")
    viz.plot_metrics_plotly()
    
    # Get summary statistics
    print("Summary Statistics:")
    summary = viz.get_summary_stats()
    print(summary)
    
    # Compare specific metrics (example)
    # viz.compare_runs('loss')
    
    # Export data
    viz.export_data()

if __name__ == "__main__":
    main()
