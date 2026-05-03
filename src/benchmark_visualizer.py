"""
GPU Tokenization Benchmark Suite
Runs benchmarks and creates visualizations comparing GPU vs tiktoken performance
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from pathlib import Path

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def run_cuda_benchmark(mode='tiktoken'):
    """
    Run the CUDA benchmark executable
    
    Args:
        mode: 'quick' or 'tiktoken' for comprehensive benchmark
    
    Returns:
        Path to the generated CSV file
    """
    print(f"Running CUDA benchmark in {mode} mode...")
    
    executable = "./vocab_lookup.exe"
    
    # Check if executable exists
    if not Path(executable).exists():
        print(f"Error: {executable} not found. Please compile with 'make' first.")
        sys.exit(1)
    
    # Check if data exists
    vocab_path = Path("data/sample_vocab.csv")
    text_path = Path("data/sample_text.txt")
    
    if not vocab_path.exists() or not text_path.exists():
        print("Test data not found. Generating...")
        subprocess.run([executable, "--generate"], check=True)
    
    # Run benchmark
    if mode == 'tiktoken':
        subprocess.run([executable, "--tiktoken"], check=True)
        csv_file = "tiktoken_benchmark_results.csv"
    else:
        subprocess.run([executable], check=True)
        csv_file = "benchmark_results.csv"
    
    print(f"Benchmark complete. Results saved to {csv_file}")
    return csv_file


def load_benchmark_results(csv_file):
    """Load benchmark results from CSV"""
    df = pd.read_csv(csv_file)
    print(f"\nLoaded {len(df)} benchmark results")
    print(df.to_string(index=False))
    return df


def plot_throughput_comparison(df, output_file='throughput_comparison.png'):
    """
    Create throughput comparison plot (MB/s)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: MB/s comparison
    x = range(len(df))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], df['cpu_mbps'], width, 
            label='CPU', color='#3498db', alpha=0.8)
    ax1.bar([i + width/2 for i in x], df['gpu_mbps'], width,
            label='GPU', color='#e74c3c', alpha=0.8)
    
    # Add tiktoken baseline line
    tiktoken_baseline = 4.0  # 3-5 MB/s midpoint
    ax1.axhline(y=tiktoken_baseline, color='green', linestyle='--', 
                linewidth=2, label='tiktoken baseline (~4 MB/s)')
    
    ax1.set_xlabel('Text Size', fontweight='bold')
    ax1.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax1.set_title('Throughput Comparison: GPU vs CPU vs tiktoken', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{size:.1f}KB" for size in df['text_size_kb']])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tokens/sec comparison
    ax2.bar([i - width/2 for i in x], df['cpu_tokens_per_sec'], width,
            label='CPU', color='#3498db', alpha=0.8)
    ax2.bar([i + width/2 for i in x], df['gpu_tokens_per_sec'], width,
            label='GPU', color='#e74c3c', alpha=0.8)
    
    # Add tiktoken baseline
    tiktoken_tokens = 100000  # 50-200k tokens/sec midpoint
    ax2.axhline(y=tiktoken_tokens, color='green', linestyle='--',
                linewidth=2, label='tiktoken baseline (~100k tokens/s)')
    
    ax2.set_xlabel('Text Size', fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax2.set_title('Token Processing Rate', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{size:.1f}KB" for size in df['text_size_kb']])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y/1000)}K'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved throughput comparison to {output_file}")
    plt.close()


def plot_speedup_analysis(df, output_file='speedup_analysis.png'):
    """
    Create speedup analysis plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Speedup vs text size
    ax1.plot(df['text_size_kb'], df['speedup'], 'o-', 
             linewidth=2, markersize=8, color='#e74c3c')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax1.fill_between(df['text_size_kb'], 1, df['speedup'], 
                     alpha=0.3, color='#e74c3c')
    
    ax1.set_xlabel('Text Size (KB)', fontweight='bold')
    ax1.set_ylabel('Speedup (GPU vs CPU)', fontweight='bold')
    ax1.set_title('GPU Speedup vs Text Size', fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add annotations for speedup values
    for i, row in df.iterrows():
        ax1.annotate(f'{row["speedup"]:.1f}x',
                    xy=(row['text_size_kb'], row['speedup']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color='darkred')
    
    # Plot 2: GPU vs tiktoken comparison (MB/s)
    tiktoken_baseline = 4.0
    gpu_vs_tiktoken = df['gpu_mbps'] / tiktoken_baseline
    
    colors = ['green' if x > 1 else 'red' for x in gpu_vs_tiktoken]
    bars = ax2.bar(range(len(df)), gpu_vs_tiktoken, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                label='tiktoken baseline')
    
    ax2.set_xlabel('Text Size', fontweight='bold')
    ax2.set_ylabel('Speedup vs tiktoken', fontweight='bold')
    ax2.set_title('GPU Performance vs tiktoken Baseline', fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f"{size:.1f}KB" for size in df['text_size_kb']])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, gpu_vs_tiktoken)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}x',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved speedup analysis to {output_file}")
    plt.close()


def plot_scaling_analysis(df, output_file='scaling_analysis.png'):
    """
    Analyze how performance scales with problem size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Time scaling
    ax1.loglog(df['text_size_kb'], df['cpu_time_ms'], 'o-',
              label='CPU', linewidth=2, markersize=8, color='#3498db')
    ax1.loglog(df['text_size_kb'], df['gpu_time_ms'], 'o-',
              label='GPU', linewidth=2, markersize=8, color='#e74c3c')
    
    ax1.set_xlabel('Text Size (KB)', fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontweight='bold')
    ax1.set_title('Execution Time Scaling', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Efficiency (tokens/ms)
    cpu_efficiency = df['num_tokens'] / df['cpu_time_ms']
    gpu_efficiency = df['num_tokens'] / df['gpu_time_ms']
    
    ax2.plot(df['text_size_kb'], cpu_efficiency, 'o-',
            label='CPU', linewidth=2, markersize=8, color='#3498db')
    ax2.plot(df['text_size_kb'], gpu_efficiency, 'o-',
            label='GPU', linewidth=2, markersize=8, color='#e74c3c')
    
    ax2.set_xlabel('Text Size (KB)', fontweight='bold')
    ax2.set_ylabel('Efficiency (tokens/ms)', fontweight='bold')
    ax2.set_title('Processing Efficiency', fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved scaling analysis to {output_file}")
    plt.close()


def create_summary_table(df, output_file='benchmark_summary.png'):
    """
    Create a summary table visualization
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary data
    summary_data = []
    tiktoken_baseline_mbps = 4.0
    tiktoken_baseline_tokens = 100000
    
    for _, row in df.iterrows():
        summary_data.append([
            f"{row['text_size_kb']:.1f} KB",
            f"{row['num_tokens']:,}",
            f"{row['cpu_mbps']:.2f}",
            f"{row['gpu_mbps']:.2f}",
            f"{row['gpu_mbps']/tiktoken_baseline_mbps:.2f}x",
            f"{row['speedup']:.2f}x",
            "✓" if row['correct'] else "✗"
        ])
    
    # Create table
    table = ax.table(cellText=summary_data,
                    colLabels=['Text Size', 'Tokens', 'CPU (MB/s)', 
                              'GPU (MB/s)', 'vs tiktoken', 'Speedup', 'Correct'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.12, 0.15, 0.15, 0.15, 0.12, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(summary_data) + 1):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Benchmark Summary: GPU Tokenization vs CPU vs tiktoken',
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary table to {output_file}")
    plt.close()


def generate_performance_report(df, output_file='performance_report.txt'):
    """
    Generate a text-based performance report
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GPU TOKENIZATION BENCHMARK REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("COMPARISON TARGET:\n")
        f.write("  tiktoken (OpenAI) single-threaded baseline:\n")
        f.write("    - Throughput: 3-5 MB/s\n")
        f.write("    - Token rate: 50-200k tokens/sec\n")
        f.write("    - Implementation: Rust core with Python bindings\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        avg_speedup = df['speedup'].mean()
        max_speedup = df['speedup'].max()
        avg_gpu_mbps = df['gpu_mbps'].mean()
        avg_gpu_tokens = df['gpu_tokens_per_sec'].mean()
        
        f.write(f"Average GPU Speedup: {avg_speedup:.2f}x over CPU\n")
        f.write(f"Maximum GPU Speedup: {max_speedup:.2f}x over CPU\n")
        f.write(f"Average GPU Throughput: {avg_gpu_mbps:.2f} MB/s\n")
        f.write(f"Average Token Rate: {avg_gpu_tokens:,.0f} tokens/sec\n\n")
        
        # Comparison to tiktoken
        tiktoken_mbps = 4.0
        tiktoken_tokens = 100000
        
        f.write("GPU vs tiktoken Comparison:\n")
        for _, row in df.iterrows():
            speedup_vs_tiktoken = row['gpu_mbps'] / tiktoken_mbps
            status = "FASTER" if speedup_vs_tiktoken > 1 else "SLOWER"
            f.write(f"  {row['text_size_kb']:7.1f} KB: "
                   f"{speedup_vs_tiktoken:5.2f}x {status} than tiktoken "
                   f"({row['gpu_mbps']:.2f} MB/s vs {tiktoken_mbps} MB/s)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for i, row in df.iterrows():
            f.write(f"Test {i+1}: {row['text_size_kb']:.1f} KB, "
                   f"{row['num_tokens']:,} tokens\n")
            f.write(f"  CPU: {row['cpu_time_ms']:.2f} ms, "
                   f"{row['cpu_mbps']:.2f} MB/s, "
                   f"{row['cpu_tokens_per_sec']:,.0f} tokens/sec\n")
            f.write(f"  GPU: {row['gpu_time_ms']:.2f} ms, "
                   f"{row['gpu_mbps']:.2f} MB/s, "
                   f"{row['gpu_tokens_per_sec']:,.0f} tokens/sec\n")
            f.write(f"  Speedup: {row['speedup']:.2f}x\n")
            f.write(f"  Correctness: {'PASS' if row['correct'] else 'FAIL'}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("="*80 + "\n\n")
        
        # Analysis
        if avg_speedup > 5:
            f.write("✓ GPU shows EXCELLENT speedup (>5x) over CPU baseline\n")
        elif avg_speedup > 2:
            f.write("✓ GPU shows GOOD speedup (2-5x) over CPU baseline\n")
        else:
            f.write("⚠ GPU shows MODEST speedup (<2x) - optimization recommended\n")
        
        if avg_gpu_mbps > tiktoken_mbps:
            f.write(f"✓ GPU OUTPERFORMS tiktoken by "
                   f"{avg_gpu_mbps/tiktoken_mbps:.2f}x on average\n")
        else:
            f.write(f"⚠ GPU underperforms tiktoken by "
                   f"{tiktoken_mbps/avg_gpu_mbps:.2f}x on average\n")
        
        if all(df['correct']):
            f.write("✓ All correctness checks PASSED\n")
        else:
            f.write("✗ Some correctness checks FAILED - investigate bugs\n")
    
    print(f"Saved performance report to {output_file}")


def main():
    """Main benchmark and visualization pipeline"""
    print("="*80)
    print("GPU Tokenization Benchmark Suite")
    print("="*80)
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        mode = 'quick'
        print("Running in QUICK mode (single iteration)\n")
    else:
        mode = 'tiktoken'
        print("Running in COMPREHENSIVE mode (100 iterations per size)\n")
    
    # Run CUDA benchmark
    csv_file = run_cuda_benchmark(mode)
    
    # Load results
    print("\n" + "="*80)
    df = load_benchmark_results(csv_file)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    plot_throughput_comparison(df)
    plot_speedup_analysis(df)
    plot_scaling_analysis(df)
    create_summary_table(df)
    generate_performance_report(df)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - throughput_comparison.png")
    print("  - speedup_analysis.png")
    print("  - scaling_analysis.png")
    print("  - benchmark_summary.png")
    print("  - performance_report.txt")
    print(f"  - {csv_file}")
    print("\nOpen the PNG files to view visualizations!")


if __name__ == "__main__":
    main()
