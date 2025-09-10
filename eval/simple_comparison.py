#!/usr/bin/env python3
"""
Simple performance comparison script for PyTorch Eager vs Triton implementations.
Creates bar plots comparing tokens_per_second and peak_memory_MB.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the data
    eager_df = pd.read_csv('eval/pytorch_model_eager')
    triton_df = pd.read_csv('eval/pytorch_model_triton2')
    
    # Add implementation type
    eager_df['implementation'] = 'Eager'
    triton_df['implementation'] = 'Triton'
    
    # Combine data
    df = pd.concat([eager_df, triton_df], ignore_index=True)
    df['config'] = df['batch_size'].astype(str) + 'x' + df['input_seq_len'].astype(str)
    
    # Filter out failed runs (tokens_per_second = 0)
    df_valid = df[df['tokens_per_second'] > 0].copy()
    
    print("Performance Comparison Summary:")
    print("=" * 50)
    
    # Create side-by-side bar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get configurations and implementations
    # Sort configurations by batch_size first, then by input_seq_len
    configs = sorted(df_valid['config'].unique(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    implementations = df_valid['implementation'].unique()
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Plot 1: Tokens per Second
    for i, impl in enumerate(implementations):
        impl_data = df_valid[df_valid['implementation'] == impl]
        tokens_per_sec = []
        
        for config in configs:
            config_data = impl_data[impl_data['config'] == config]
            if not config_data.empty:
                tokens_per_sec.append(config_data['tokens_per_second'].iloc[0])
            else:
                tokens_per_sec.append(0)
        
        ax1.bar(x + i * width, tokens_per_sec, width, label=impl, alpha=0.8)
    
    ax1.set_xlabel('Configuration (Batch × Seq Length)')
    ax1.set_ylabel('Tokens per Second')
    ax1.set_title('Performance: Tokens per Second')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Peak Memory Usage
    for i, impl in enumerate(implementations):
        impl_data = df_valid[df_valid['implementation'] == impl]
        memory_usage = []
        
        for config in configs:
            config_data = impl_data[impl_data['config'] == config]
            if not config_data.empty:
                memory_usage.append(config_data['peak_memory_MB'].iloc[0])
            else:
                memory_usage.append(0)
        
        ax2.bar(x + i * width, memory_usage, width, label=impl, alpha=0.8)
    
    ax2.set_xlabel('Configuration (Batch × Seq Length)')
    ax2.set_ylabel('Peak Memory Usage (MB)')
    ax2.set_title('Memory Usage: Peak Memory')
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison table
    print("\nDetailed Performance Data:")
    print("-" * 80)
    
    for config in configs:
        eager_data = df_valid[(df_valid['config'] == config) & (df_valid['implementation'] == 'Eager')]
        triton_data = df_valid[(df_valid['config'] == config) & (df_valid['implementation'] == 'Triton')]
        
        if not eager_data.empty and not triton_data.empty:
            eager_tps = eager_data['tokens_per_second'].iloc[0]
            triton_tps = triton_data['tokens_per_second'].iloc[0]
            eager_mem = eager_data['peak_memory_MB'].iloc[0]
            triton_mem = triton_data['peak_memory_MB'].iloc[0]
            speedup = triton_tps / eager_tps
            
            print(f"Config {config}:")
            print(f"  Eager  - TPS: {eager_tps:6.2f}, Memory: {eager_mem:8.2f} MB")
            print(f"  Triton - TPS: {triton_tps:6.2f}, Memory: {triton_mem:8.2f} MB")
            print(f"  Speedup: {speedup:.2f}×")
            print()
        elif not triton_data.empty:
            triton_tps = triton_data['tokens_per_second'].iloc[0]
            triton_mem = triton_data['peak_memory_MB'].iloc[0]
            print(f"Config {config}:")
            print(f"  Eager  - FAILED")
            print(f"  Triton - TPS: {triton_tps:6.2f}, Memory: {triton_mem:8.2f} MB")
            print(f"  Speedup: N/A (Eager failed)")
            print()

if __name__ == "__main__":
    main()
