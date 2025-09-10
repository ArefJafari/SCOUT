#!/usr/bin/env python3
"""
Performance comparison script for PyTorch Eager vs Triton implementations.
Creates bar plots comparing tokens_per_second and peak_memory_MB across different configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_data(eager_file, triton_file):
    """Load and process the CSV data from both files."""
    # Load the data
    eager_df = pd.read_csv(eager_file)
    triton_df = pd.read_csv(triton_file)
    
    # Add implementation type column
    eager_df['implementation'] = 'Eager'
    triton_df['implementation'] = 'Triton'
    
    # Combine the dataframes
    combined_df = pd.concat([eager_df, triton_df], ignore_index=True)
    
    # Create a configuration label for better plotting
    combined_df['config'] = combined_df['batch_size'].astype(str) + 'x' + combined_df['input_seq_len'].astype(str)
    
    return combined_df

def plot_tokens_per_second(df, save_path=None):
    """Create bar plot comparing tokens per second."""
    # Filter out rows where tokens_per_second is 0 (failed runs)
    df_filtered = df[df['tokens_per_second'] > 0].copy()
    
    if df_filtered.empty:
        print("No valid tokens_per_second data to plot")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique configurations and implementations
    # Sort configurations by batch_size first, then by input_seq_len
    configs = sorted(df_filtered['config'].unique(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    implementations = df_filtered['implementation'].unique()
    
    # Set up bar positions
    x = np.arange(len(configs))
    width = 0.35
    
    # Plot bars for each implementation
    for i, impl in enumerate(implementations):
        impl_data = df_filtered[df_filtered['implementation'] == impl]
        tokens_per_sec = []
        
        for config in configs:
            config_data = impl_data[impl_data['config'] == config]
            if not config_data.empty:
                tokens_per_sec.append(config_data['tokens_per_second'].iloc[0])
            else:
                tokens_per_sec.append(0)
        
        ax.bar(x + i * width, tokens_per_sec, width, label=impl, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Configuration (Batch Size × Input Sequence Length)')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Performance Comparison: Tokens per Second\nEager vs Triton Implementation')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, impl in enumerate(implementations):
        impl_data = df_filtered[df_filtered['implementation'] == impl]
        for j, config in enumerate(configs):
            config_data = impl_data[impl_data['config'] == config]
            if not config_data.empty:
                value = config_data['tokens_per_second'].iloc[0]
                ax.text(j + i * width, value + 0.5, f'{value:.1f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tokens per second plot saved to: {save_path}")
    
    plt.show()

def plot_peak_memory(df, save_path=None):
    """Create bar plot comparing peak memory usage."""
    # Filter out rows where peak_memory_MB is 0 (failed runs)
    df_filtered = df[df['peak_memory_MB'] > 0].copy()
    
    if df_filtered.empty:
        print("No valid peak_memory_MB data to plot")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique configurations and implementations
    # Sort configurations by batch_size first, then by input_seq_len
    configs = sorted(df_filtered['config'].unique(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    implementations = df_filtered['implementation'].unique()
    
    # Set up bar positions
    x = np.arange(len(configs))
    width = 0.35
    
    # Plot bars for each implementation
    for i, impl in enumerate(implementations):
        impl_data = df_filtered[df_filtered['implementation'] == impl]
        memory_usage = []
        
        for config in configs:
            config_data = impl_data[impl_data['config'] == config]
            if not config_data.empty:
                memory_usage.append(config_data['peak_memory_MB'].iloc[0])
            else:
                memory_usage.append(0)
        
        ax.bar(x + i * width, memory_usage, width, label=impl, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Configuration (Batch Size × Input Sequence Length)')
    ax.set_ylabel('Peak Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison: Peak Memory\nEager vs Triton Implementation')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, impl in enumerate(implementations):
        impl_data = df_filtered[df_filtered['implementation'] == impl]
        for j, config in enumerate(configs):
            config_data = impl_data[impl_data['config'] == config]
            if not config_data.empty:
                value = config_data['peak_memory_MB'].iloc[0]
                ax.text(j + i * width, value + 50, f'{value:.0f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Peak memory plot saved to: {save_path}")
    
    plt.show()

def plot_speedup_comparison(df, save_path=None):
    """Create a plot showing speedup ratio (Triton/Eager) for each configuration."""
    # Filter out rows where tokens_per_second is 0 (failed runs)
    df_filtered = df[df['tokens_per_second'] > 0].copy()
    
    if df_filtered.empty:
        print("No valid data for speedup comparison")
        return
    
    # Calculate speedup for each configuration
    # Sort configurations by batch_size first, then by input_seq_len
    configs = sorted(df_filtered['config'].unique(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    speedups = []
    
    for config in configs:
        eager_data = df_filtered[(df_filtered['config'] == config) & (df_filtered['implementation'] == 'Eager')]
        triton_data = df_filtered[(df_filtered['config'] == config) & (df_filtered['implementation'] == 'Triton')]
        
        if not eager_data.empty and not triton_data.empty:
            eager_tps = eager_data['tokens_per_second'].iloc[0]
            triton_tps = triton_data['tokens_per_second'].iloc[0]
            speedup = triton_tps / eager_tps if eager_tps > 0 else 0
            speedups.append(speedup)
        else:
            speedups.append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(configs, speedups, alpha=0.8, color='green')
    
    # Add horizontal line at 1.0 (no speedup)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Speedup (1.0×)')
    
    # Customize the plot
    ax.set_xlabel('Configuration (Batch Size × Input Sequence Length)')
    ax.set_ylabel('Speedup Ratio (Triton/Eager)')
    ax.set_title('Speedup Comparison: Triton vs Eager Implementation')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{speedup:.2f}×', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Speedup comparison plot saved to: {save_path}")
    
    plt.show()

def print_summary_table(df):
    """Print a summary table of the performance comparison."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    # Filter out failed runs
    df_filtered = df[df['tokens_per_second'] > 0].copy()
    
    if df_filtered.empty:
        print("No valid performance data found")
        return
    
    # Group by configuration and implementation
    summary = df_filtered.groupby(['config', 'implementation']).agg({
        'tokens_per_second': 'first',
        'peak_memory_MB': 'first'
    }).reset_index()
    
    # Pivot to have implementations as columns
    pivot_tps = summary.pivot(index='config', columns='implementation', values='tokens_per_second')
    pivot_mem = summary.pivot(index='config', columns='implementation', values='peak_memory_MB')
    
    print("\nTokens per Second:")
    print("-" * 50)
    print(pivot_tps.round(2))
    
    print("\nPeak Memory Usage (MB):")
    print("-" * 50)
    print(pivot_mem.round(2))
    
    # Calculate speedup
    if 'Eager' in pivot_tps.columns and 'Triton' in pivot_tps.columns:
        speedup = (pivot_tps['Triton'] / pivot_tps['Eager']).round(2)
        print("\nSpeedup (Triton/Eager):")
        print("-" * 50)
        print(speedup)
        
        print(f"\nAverage Speedup: {speedup.mean():.2f}×")
        print(f"Maximum Speedup: {speedup.max():.2f}×")
        print(f"Minimum Speedup: {speedup.min():.2f}×")

def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch Eager vs Triton performance')
    parser.add_argument('--eager_file', default='eval/pytorch_model_eager', 
                       help='Path to eager implementation results CSV')
    parser.add_argument('--triton_file', default='eval/pytorch_model_triton2', 
                       help='Path to triton implementation results CSV')
    parser.add_argument('--save_plots', action='store_true', 
                       help='Save plots to files')
    parser.add_argument('--output_dir', default='plots', 
                       help='Directory to save plots (if --save_plots is used)')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.eager_file):
        print(f"Error: Eager file not found: {args.eager_file}")
        return
    
    if not os.path.exists(args.triton_file):
        print(f"Error: Triton file not found: {args.triton_file}")
        return
    
    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading performance data...")
    df = load_data(args.eager_file, args.triton_file)
    
    # Print summary table
    print_summary_table(df)
    
    # Create plots
    print("\nGenerating plots...")
    
    # Tokens per second comparison
    tps_save_path = os.path.join(args.output_dir, 'tokens_per_second_comparison.png') if args.save_plots else None
    plot_tokens_per_second(df, tps_save_path)
    
    # Peak memory comparison
    mem_save_path = os.path.join(args.output_dir, 'peak_memory_comparison.png') if args.save_plots else None
    plot_peak_memory(df, mem_save_path)
    
    # Speedup comparison
    speedup_save_path = os.path.join(args.output_dir, 'speedup_comparison.png') if args.save_plots else None
    plot_speedup_comparison(df, speedup_save_path)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
