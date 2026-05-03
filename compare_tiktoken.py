#!/usr/bin/env python3
"""
Direct tiktoken comparison benchmark
Runs tiktoken on the same data and compares with GPU results
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")


def benchmark_tiktoken(text, encoding_name="cl100k_base", num_iterations=100):
    """
    Benchmark tiktoken encoding performance
    
    Args:
        text: Input text string
        encoding_name: tiktoken encoding to use
        num_iterations: Number of iterations to average
    
    Returns:
        dict with benchmark results
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    
    enc = tiktoken.get_encoding(encoding_name)
    
    # Warmup
    for _ in range(10):
        tokens = enc.encode(text)
    
    # Actual benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        tokens = enc.encode(text)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / num_iterations) * 1000
    text_bytes = len(text.encode('utf-8'))
    
    return {
        'time_ms': avg_time_ms,
        'num_tokens': len(tokens),
        'text_bytes': text_bytes,
        'tokens_per_sec': len(tokens) / (avg_time_ms / 1000),
        'mbps': (text_bytes / (1024**2)) / (avg_time_ms / 1000)
    }


def load_text_file(filepath, max_size=None):
    """Load text file for benchmarking"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if max_size and len(text) > max_size:
        text = text[:max_size]
    
    return text


def run_tiktoken_benchmark_suite(text_path="data/sample_text.txt"):
    """
    Run comprehensive tiktoken benchmark on multiple text sizes
    """
    if not TIKTOKEN_AVAILABLE:
        print("tiktoken not available. Skipping benchmark.")
        return None
    
    print("="*80)
    print("tiktoken Benchmark Suite")
    print("="*80)
    print()
    
    # Load full text
    full_text = load_text_file(text_path)
    
    # Test sizes matching CUDA benchmark
    test_sizes = [
        ("Short (1KB)", 1024),
        ("Medium (10KB)", 10*1024),
        ("Long (100KB)", 100*1024),
        ("XLong (1MB)", 1024*1024)
    ]
    
    results = []
    
    print(f"{'Size':<20} | {'Tokens':>10} | {'Time (ms)':>12} | "
          f"{'Tokens/sec':>15} | {'MB/s':>10}")
    print("-" * 80)
    
    for label, size in test_sizes:
        text = full_text[:min(size, len(full_text))]
        result = benchmark_tiktoken(text, num_iterations=100)
        
        if result:
            print(f"{label:<20} | {result['num_tokens']:>10,} | "
                  f"{result['time_ms']:>12.2f} | "
                  f"{result['tokens_per_sec']:>15,.0f} | "
                  f"{result['mbps']:>10.2f}")
            
            results.append({
                'text_size': label,
                'text_size_kb': result['text_bytes'] / 1024,
                **result
            })
    
    return pd.DataFrame(results)


def compare_with_cuda_results(tiktoken_df, cuda_csv="tiktoken_benchmark_results.csv"):
    """
    Compare tiktoken results with CUDA benchmark results
    """
    if not Path(cuda_csv).exists():
        print(f"Error: {cuda_csv} not found. Run CUDA benchmark first.")
        return
    
    cuda_df = pd.read_csv(cuda_csv)
    
    # Create comparison DataFrame
    comparison = []
    
    for i in range(min(len(tiktoken_df), len(cuda_df))):
        tiktoken_row = tiktoken_df.iloc[i]
        cuda_row = cuda_df.iloc[i]
        
        comparison.append({
            'text_size': tiktoken_row['text_size'],
            'text_size_kb': cuda_row['text_size_kb'],
            'num_tokens': cuda_row['num_tokens'],
            'tiktoken_mbps': tiktoken_row['mbps'],
            'cpu_mbps': cuda_row['cpu_mbps'],
            'gpu_mbps': cuda_row['gpu_mbps'],
            'gpu_vs_tiktoken': cuda_row['gpu_mbps'] / tiktoken_row['mbps'],
            'cpu_vs_tiktoken': cuda_row['cpu_mbps'] / tiktoken_row['mbps'],
            'tiktoken_tokens_per_sec': tiktoken_row['tokens_per_sec'],
            'gpu_tokens_per_sec': cuda_row['gpu_tokens_per_sec']
        })
    
    comp_df = pd.DataFrame(comparison)
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON: GPU vs tiktoken")
    print("="*80)
    print()
    print(comp_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    # Save to CSV
    comp_df.to_csv("gpu_vs_tiktoken_comparison.csv", index=False)
    print("\nComparison saved to: gpu_vs_tiktoken_comparison.csv")
    
    return comp_df


def plot_tiktoken_comparison(comp_df, output_file='tiktoken_vs_gpu.png'):
    """
    Create visualization comparing tiktoken, CPU, and GPU
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x = range(len(comp_df))
    width = 0.25
    
    # Plot 1: Throughput (MB/s) comparison
    ax1.bar([i - width for i in x], comp_df['tiktoken_mbps'], width,
           label='tiktoken', color='#2ecc71', alpha=0.8)
    ax1.bar([i for i in x], comp_df['cpu_mbps'], width,
           label='Our CPU', color='#3498db', alpha=0.8)
    ax1.bar([i + width for i in x], comp_df['gpu_mbps'], width,
           label='Our GPU', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Text Size', fontweight='bold')
    ax1.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax1.set_title('Throughput Comparison: GPU vs CPU vs tiktoken', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_df['text_size'], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Tokens/sec comparison
    ax2.bar([i - width for i in x], comp_df['tiktoken_tokens_per_sec'], width,
           label='tiktoken', color='#2ecc71', alpha=0.8)
    ax2.bar([i + width for i in x], comp_df['gpu_tokens_per_sec'], width,
           label='Our GPU', color='#e74c3c', alpha=0.8)
    
    ax2.set_xlabel('Text Size', fontweight='bold')
    ax2.set_ylabel('Tokens/sec', fontweight='bold')
    ax2.set_title('Token Processing Rate', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_df['text_size'], rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y/1000)}K'))
    
    # Plot 3: GPU speedup vs tiktoken
    colors_gpu = ['green' if x > 1 else 'red' for x in comp_df['gpu_vs_tiktoken']]
    bars_gpu = ax3.bar(x, comp_df['gpu_vs_tiktoken'], color=colors_gpu, alpha=0.7)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
               label='tiktoken baseline')
    
    ax3.set_xlabel('Text Size', fontweight='bold')
    ax3.set_ylabel('Speedup', fontweight='bold')
    ax3.set_title('GPU Speedup vs tiktoken Baseline', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(comp_df['text_size'], rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars_gpu, comp_df['gpu_vs_tiktoken']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Scaling comparison
    ax4.plot(comp_df['text_size_kb'], comp_df['tiktoken_mbps'], 'o-',
            label='tiktoken', linewidth=2, markersize=8, color='#2ecc71')
    ax4.plot(comp_df['text_size_kb'], comp_df['gpu_mbps'], 'o-',
            label='Our GPU', linewidth=2, markersize=8, color='#e74c3c')
    ax4.fill_between(comp_df['text_size_kb'], 
                     comp_df['tiktoken_mbps'], 
                     comp_df['gpu_mbps'],
                     alpha=0.2, color='green')
    
    ax4.set_xlabel('Text Size (KB)', fontweight='bold')
    ax4.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax4.set_title('Performance Scaling', fontweight='bold')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved tiktoken comparison plot to {output_file}")
    plt.close()


def generate_tiktoken_report(comp_df, output_file='tiktoken_comparison_report.txt'):
    """
    Generate detailed tiktoken comparison report
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GPU vs tiktoken COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("BASELINE:\n")
        f.write("  tiktoken (OpenAI's BPE tokenizer)\n")
        f.write("  - Implementation: Rust core with Python bindings\n")
        f.write("  - Encoding: cl100k_base (GPT-4)\n")
        f.write("  - Single-threaded performance\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for _, row in comp_df.iterrows():
            f.write(f"{row['text_size']}:\n")
            f.write(f"  tiktoken: {row['tiktoken_mbps']:.2f} MB/s, "
                   f"{row['tiktoken_tokens_per_sec']:,.0f} tokens/sec\n")
            f.write(f"  Our GPU:  {row['gpu_mbps']:.2f} MB/s, "
                   f"{row['gpu_tokens_per_sec']:,.0f} tokens/sec\n")
            f.write(f"  Speedup:  {row['gpu_vs_tiktoken']:.2f}x "
                   f"{'FASTER' if row['gpu_vs_tiktoken'] > 1 else 'SLOWER'}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        avg_speedup = comp_df['gpu_vs_tiktoken'].mean()
        max_speedup = comp_df['gpu_vs_tiktoken'].max()
        
        f.write(f"Average GPU speedup vs tiktoken: {avg_speedup:.2f}x\n")
        f.write(f"Maximum GPU speedup vs tiktoken: {max_speedup:.2f}x\n")
        f.write(f"GPU faster on: {sum(comp_df['gpu_vs_tiktoken'] > 1)}/{len(comp_df)} tests\n\n")
        
        if avg_speedup > 1:
            f.write(f"✓ GPU implementation is {avg_speedup:.2f}x FASTER than tiktoken on average\n")
        else:
            f.write(f"⚠ GPU implementation is {1/avg_speedup:.2f}x SLOWER than tiktoken on average\n")
    
    print(f"Saved tiktoken comparison report to {output_file}")


def main():
    """Main tiktoken comparison pipeline"""
    print("="*80)
    print("tiktoken vs GPU Benchmark Comparison")
    print("="*80)
    print()
    
    if not TIKTOKEN_AVAILABLE:
        print("Error: tiktoken not installed.")
        print("Install with: pip install tiktoken")
        print("\nOr run: pip install -r requirements.txt")
        return
    
    # Check if test data exists
    text_path = "data/sample_text.txt"
    if not Path(text_path).exists():
        print(f"Error: {text_path} not found.")
        print("Run: ./vocab_lookup.exe --generate")
        return
    
    # Run tiktoken benchmark
    tiktoken_df = run_tiktoken_benchmark_suite(text_path)
    
    if tiktoken_df is None:
        return
    
    # Save tiktoken results
    tiktoken_df.to_csv("tiktoken_results.csv", index=False)
    print("\ntiktoken results saved to: tiktoken_results.csv")
    
    # Compare with CUDA results
    print("\n" + "="*80)
    print("Comparing with CUDA benchmark results...")
    print("="*80)
    
    comp_df = compare_with_cuda_results(tiktoken_df)
    
    if comp_df is not None:
        # Generate visualizations
        plot_tiktoken_comparison(comp_df)
        generate_tiktoken_report(comp_df)
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - tiktoken_results.csv")
        print("  - gpu_vs_tiktoken_comparison.csv")
        print("  - tiktoken_vs_gpu.png")
        print("  - tiktoken_comparison_report.txt")


if __name__ == "__main__":
    main()
