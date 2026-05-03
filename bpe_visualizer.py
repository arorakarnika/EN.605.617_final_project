"""
Visualize BPE benchmark results from the CUDA sweep CSV.

Workflow
--------
1. (Optional) Generate fresh sweep data:
       ./vocab_lookup.exe --bpe --sweep --csv data/bpe_benchmark.csv --iters 100

2. Measure tiktoken's own throughput on the same pieces.bin so we have a
   baseline to compare against (kernel-only time vs full Python call).

3. Render two plots and a text summary:
       throughput_vs_threads.png : V1 and V2 MB/s + tokens/sec across thread
                                   counts.
       speedup_vs_tiktoken.png   : speedup of every GPU config over the
                                   tiktoken Python baseline.
       bpe_report.txt            : short text report mirroring the plots.
"""

import argparse
import struct
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tiktoken

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10


def load_pieces(path):
    with open(path, "rb") as f:
        (num_pieces, total_bytes) = struct.unpack("<II", f.read(8))
        offsets = []
        lengths = []
        for _ in range(num_pieces):
            off, ln = struct.unpack("<II", f.read(8))
            offsets.append(off)
            lengths.append(ln)
        blob = f.read(total_bytes)
    pieces = [blob[offsets[i]:offsets[i] + lengths[i]] for i in range(num_pieces)]
    return pieces, total_bytes


def measure_tiktoken_baseline(pieces, total_bytes, encoding_name, iterations):
    enc = tiktoken.get_encoding(encoding_name)
    enc._encode_single_piece(pieces[0]) if pieces else None  # warm cache

    timings = []
    total_tokens = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        local_total = 0
        for piece_bytes in pieces:
            ids = enc._encode_single_piece(piece_bytes)
            local_total += len(ids)
        timings.append(time.perf_counter() - t0)
        total_tokens = local_total

    avg_sec = sum(timings) / len(timings)
    mbps = (total_bytes / (1024.0 * 1024.0)) / avg_sec
    tps = total_tokens / avg_sec
    return {
        "avg_time_ms": avg_sec * 1000.0,
        "mbps": mbps,
        "tokens_per_sec": tps,
        "num_tokens": total_tokens,
    }


def maybe_run_sweep(executable, csv_path, iters, ranks_path, pieces_path):
    if Path(csv_path).exists() and Path(csv_path).stat().st_size > 0:
        print(f"Reusing existing CSV: {csv_path}")
        return
    if not Path(executable).exists():
        raise SystemExit(f"Executable {executable} not found - run 'make' first")
    print(f"Running sweep: {executable} --bpe --sweep --csv {csv_path}")
    subprocess.run(
        [
            executable,
            "--bpe",
            "--sweep",
            "--csv",
            csv_path,
            "--iters",
            str(iters),
            "--ranks",
            ranks_path,
            "--pieces",
            pieces_path,
        ],
        check=True,
    )


def plot_throughput_vs_threads(df, baseline, output_path):
    fig, (ax_mbps, ax_tps) = plt.subplots(1, 2, figsize=(14, 5))

    for kernel, color in [("v1", "#3498db"), ("v2", "#e74c3c")]:
        sub = df[df["kernel"] == kernel].sort_values("threads_per_block")
        if sub.empty:
            continue
        ax_mbps.plot(sub["threads_per_block"], sub["mbps"],
                     marker="o", label=f"GPU {kernel.upper()}", color=color, linewidth=2)
        ax_tps.plot(sub["threads_per_block"], sub["tokens_per_sec"],
                    marker="o", label=f"GPU {kernel.upper()}", color=color, linewidth=2)

    ax_mbps.axhline(baseline["mbps"], linestyle="--", color="gray",
                    label=f"tiktoken ({baseline['mbps']:.1f} MB/s)")
    ax_tps.axhline(baseline["tokens_per_sec"], linestyle="--", color="gray",
                   label=f"tiktoken ({baseline['tokens_per_sec']:.0f} tok/s)")

    ax_mbps.set_xscale("log", base=2)
    ax_tps.set_xscale("log", base=2)
    ax_mbps.set_xlabel("Threads per block")
    ax_tps.set_xlabel("Threads per block")
    ax_mbps.set_ylabel("Throughput (MB/s)")
    ax_tps.set_ylabel("Throughput (tokens/sec)")
    ax_mbps.set_title("BPE encode throughput (MB/s)")
    ax_tps.set_title("BPE encode throughput (tokens/sec)")
    ax_mbps.legend()
    ax_tps.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_speedup_vs_tiktoken(df, baseline, output_path):
    df = df.copy()
    df["speedup_mbps"] = df["mbps"] / baseline["mbps"]
    df["label"] = df.apply(
        lambda r: f"{r['kernel'].upper()} t={int(r['threads_per_block'])}", axis=1
    )
    df = df.sort_values(["kernel", "threads_per_block"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#3498db" if k == "v1" else "#e74c3c" for k in df["kernel"]]
    bars = ax.bar(df["label"], df["speedup_mbps"], color=colors, alpha=0.85)
    ax.axhline(1.0, linestyle="--", color="gray", label="tiktoken baseline")

    for bar, val in zip(bars, df["speedup_mbps"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:.1f}x", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Speedup over tiktoken (MB/s ratio)")
    ax.set_title("GPU BPE speedup vs tiktoken Python baseline")
    plt.xticks(rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def write_report(df, baseline, output_path):
    best_v1 = df[df["kernel"] == "v1"].nlargest(1, "mbps")
    best_v2 = df[df["kernel"] == "v2"].nlargest(1, "mbps")

    lines = []
    lines.append("BPE Benchmark Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("tiktoken baseline (Python):")
    lines.append(f"  avg time    : {baseline['avg_time_ms']:.3f} ms")
    lines.append(f"  throughput  : {baseline['mbps']:.2f} MB/s")
    lines.append(f"  tokens/sec  : {baseline['tokens_per_sec']:.0f}")
    lines.append(f"  num tokens  : {baseline['num_tokens']}")
    lines.append("")
    if not best_v1.empty:
        r = best_v1.iloc[0]
        lines.append(f"Best V1 config: threads={int(r['threads_per_block'])} blocks={int(r['blocks'])}")
        lines.append(f"  time={r['avg_time_ms']:.3f} ms  mbps={r['mbps']:.2f}  tok/s={r['tokens_per_sec']:.0f}")
        lines.append(f"  speedup vs tiktoken (MB/s): {r['mbps'] / baseline['mbps']:.2f}x")
        lines.append("")
    if not best_v2.empty:
        r = best_v2.iloc[0]
        lines.append(f"Best V2 config: threads={int(r['threads_per_block'])} blocks={int(r['blocks'])}")
        lines.append(f"  time={r['avg_time_ms']:.3f} ms  mbps={r['mbps']:.2f}  tok/s={r['tokens_per_sec']:.0f}")
        lines.append(f"  speedup vs tiktoken (MB/s): {r['mbps'] / baseline['mbps']:.2f}x")
        lines.append("")
    lines.append("All configurations:")
    lines.append(df.to_string(index=False))
    lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BPE sweep CSV vs tiktoken")
    parser.add_argument("--csv", default="data/bpe_benchmark.csv")
    parser.add_argument("--pieces", default="data/pieces.bin")
    parser.add_argument("--ranks", default="data/bpe_ranks.bin")
    parser.add_argument("--executable", default="./vocab_lookup.exe")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--iters", type=int, default=100,
                        help="Iterations for both the GPU sweep and the tiktoken baseline")
    parser.add_argument("--no-run", action="store_true",
                        help="Do not invoke the executable even if CSV is missing")
    args = parser.parse_args()

    if not args.no_run:
        maybe_run_sweep(args.executable, args.csv, args.iters, args.ranks, args.pieces)

    if not Path(args.csv).exists():
        raise SystemExit(f"CSV not found: {args.csv} (re-run without --no-run, or create it manually)")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"CSV is empty: {args.csv}")

    print(f"\nLoaded {len(df)} sweep rows from {args.csv}")
    print(df.to_string(index=False))

    pieces, total_bytes = load_pieces(args.pieces)
    print(f"\nMeasuring tiktoken baseline ({args.iters} iterations over {len(pieces)} pieces)...")
    baseline = measure_tiktoken_baseline(pieces, total_bytes, args.encoding, args.iters)
    print(f"  tiktoken: {baseline['mbps']:.2f} MB/s, {baseline['tokens_per_sec']:.0f} tokens/sec")

    plot_throughput_vs_threads(df, baseline, "throughput_vs_threads.png")
    plot_speedup_vs_tiktoken(df, baseline, "speedup_vs_tiktoken.png")
    write_report(df, baseline, "bpe_report.txt")


if __name__ == "__main__":
    main()
