"""
Visualize BPE benchmark results from the CUDA sweep CSV.

What is timed
-------------
GPU side (from the CSV produced by ./vocab_lookup.exe --bpe --sweep --csv ...):
    kernel_*  : kernel-only time. Inputs already on device, outputs stay on
                device. Compares directly to the per-piece tiktoken loop.
    e2e_*     : H2D pieces + kernel + D2H token IDs per iteration. The 13.6 MB
                ranks table is loaded once and reused (matches a long-running
                tokenizer service). Compares directly to enc.encode(text).

tiktoken baselines (measured by this script in Python on the SAME corpus):
    encode_full      : enc.encode(text). Single Rust call: regex pre-split +
                       per-piece BPE. This is what production users run.
    per_piece_loop   : Python loop calling enc._encode_single_piece(piece) for
                       every piece. Same BPE work, but pays Python<->Rust
                       boundary overhead once per piece.
    regex_only       : Python regex.finditer time on the raw text, reported
                       so the GPU end-to-end number can be mentally combined
                       with it for an apples-to-apples pipeline comparison.

Apples-to-apples comparisons
----------------------------
    GPU kernel_*  vs  tiktoken per_piece_loop  -> "BPE merge core" speedup
    GPU e2e_*     vs  tiktoken encode_full     -> "tokenizer service" speedup
                                                  (when GPU regex-split is
                                                   amortized or done in Rust)
"""

import argparse
import struct
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import regex
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


def time_iterations(fn, iterations):
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return sum(timings) / len(timings)


def measure_baselines(text, pieces, total_bytes, encoding_name, iterations):
    enc = tiktoken.get_encoding(encoding_name)
    pat = regex.compile(enc._pat_str)

    enc.encode_ordinary(text)
    enc._encode_single_piece(pieces[0]) if pieces else None

    avg_full = time_iterations(lambda: enc.encode_ordinary(text), iterations)

    def per_piece():
        out = []
        for pb in pieces:
            out.extend(enc._encode_single_piece(pb))
        return out
    avg_loop = time_iterations(per_piece, iterations)
    full_ids = enc.encode_ordinary(text)
    loop_ids = per_piece()
    num_tokens = len(full_ids)

    avg_regex = time_iterations(lambda: [m.group(0) for m in pat.finditer(text)],
                                iterations)

    mb = total_bytes / (1024.0 * 1024.0)
    return {
        "encode_full": {
            "avg_time_ms": avg_full * 1000.0,
            "mbps": mb / avg_full,
            "tokens_per_sec": num_tokens / avg_full,
            "num_tokens": num_tokens,
        },
        "per_piece_loop": {
            "avg_time_ms": avg_loop * 1000.0,
            "mbps": mb / avg_loop,
            "tokens_per_sec": len(loop_ids) / avg_loop,
            "num_tokens": len(loop_ids),
        },
        "regex_only": {
            "avg_time_ms": avg_regex * 1000.0,
            "mbps": mb / avg_regex,
        },
    }


def maybe_run_sweep(executable, csv_path, iters, ranks_path, pieces_path):
    if Path(csv_path).exists() and Path(csv_path).stat().st_size > 0:
        print("Reusing existing CSV: {}".format(csv_path))
        return
    if not Path(executable).exists():
        raise SystemExit("Executable {} not found - run 'make' first".format(executable))
    print("Running sweep: {} --bpe --sweep --csv {}".format(executable, csv_path))
    subprocess.run(
        [
            executable, "--bpe", "--sweep", "--csv", csv_path,
            "--iters", str(iters),
            "--ranks", ranks_path,
            "--pieces", pieces_path,
        ],
        check=True,
    )


def normalize_columns(df):
    # Backwards compatibility: older CSVs used avg_time_ms / mbps / tokens_per_sec
    # for the kernel-only run and had no e2e columns at all.
    if "avg_time_ms" in df.columns and "kernel_time_ms" not in df.columns:
        df = df.rename(columns={
            "avg_time_ms": "kernel_time_ms",
            "mbps": "kernel_mbps",
            "tokens_per_sec": "kernel_tokens_per_sec",
        })
    for col in ["e2e_time_ms", "e2e_mbps", "e2e_tokens_per_sec"]:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def add_pipeline_columns(df, baselines):
    """Combine GPU end-to-end time with Python regex pre-split time so the
    GPU side measures the same work as tiktoken.encode(): regex split + BPE
    + transfers. The result is the truly apples-to-apples pipeline number."""
    df = df.copy()
    regex_ms = baselines["regex_only"]["avg_time_ms"]
    df["pipeline_time_ms"] = df["e2e_time_ms"] + regex_ms
    pipeline_sec = df["pipeline_time_ms"] / 1000.0
    df["pipeline_mbps"] = (df["input_bytes"] / (1024.0 * 1024.0)) / pipeline_sec
    df["pipeline_tokens_per_sec"] = df["num_tokens"] / pipeline_sec
    return df


def plot_throughput_vs_threads(df, baselines, output_path):
    fig, (ax_mbps, ax_tps) = plt.subplots(1, 2, figsize=(14, 5))

    style = {
        ("v1", "kernel"):   {"color": "#3498db", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0,
                             "label": "GPU V1 kernel only"},
        ("v1", "e2e"):      {"color": "#3498db", "linestyle": "--", "linewidth": 2.0, "alpha": 0.85,
                             "label": "GPU V1 end-to-end (transfers)"},
        ("v1", "pipeline"): {"color": "#3498db", "linestyle": ":",  "linewidth": 2.5, "alpha": 0.65,
                             "label": "GPU V1 pipeline (+ Python regex)"},
        ("v2", "kernel"):   {"color": "#e74c3c", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0,
                             "label": "GPU V2 kernel only"},
        ("v2", "e2e"):      {"color": "#e74c3c", "linestyle": "--", "linewidth": 2.0, "alpha": 0.85,
                             "label": "GPU V2 end-to-end (transfers)"},
        ("v2", "pipeline"): {"color": "#e74c3c", "linestyle": ":",  "linewidth": 2.5, "alpha": 0.65,
                             "label": "GPU V2 pipeline (+ Python regex)"},
    }

    for kernel in ("v1", "v2"):
        sub = df[df["kernel"] == kernel].sort_values("threads_per_block")
        if sub.empty:
            continue
        for which, mbps_col, tps_col in [
            ("kernel", "kernel_mbps", "kernel_tokens_per_sec"),
            ("e2e", "e2e_mbps", "e2e_tokens_per_sec"),
            ("pipeline", "pipeline_mbps", "pipeline_tokens_per_sec"),
        ]:
            if mbps_col not in sub.columns or sub[mbps_col].isna().all():
                continue
            s = style[(kernel, which)]
            ax_mbps.plot(sub["threads_per_block"], sub[mbps_col], marker="o", **s)
            ax_tps.plot(sub["threads_per_block"], sub[tps_col], marker="o", **s)

    for name, info in [
        ("encode_full",    {"color": "#222222", "linestyle": "-.", "linewidth": 1.5}),
        ("per_piece_loop", {"color": "#777777", "linestyle": "-.", "linewidth": 1.5}),
    ]:
        b = baselines.get(name)
        if not b:
            continue
        label_mbps = "tiktoken {} ({:.1f} MB/s)".format(name, b["mbps"])
        label_tps = "tiktoken {} ({:.0f} tok/s)".format(name, b["tokens_per_sec"])
        ax_mbps.axhline(b["mbps"], label=label_mbps, **info)
        ax_tps.axhline(b["tokens_per_sec"], label=label_tps, **info)

    for ax in (ax_mbps, ax_tps):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads per block")
        ax.legend(fontsize=7, loc="best")
    ax_mbps.set_ylabel("Throughput (MB/s)")
    ax_tps.set_ylabel("Throughput (tokens/sec)")
    ax_mbps.set_title("BPE encode throughput - MB/s")
    ax_tps.set_title("BPE encode throughput - tokens/sec")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(output_path))


def _bar_panel(ax, df, values, baseline_mbps, baseline_label, title, colors):
    bars = ax.bar(df["label"], values, color=colors, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--",
               label="{} ({:.1f} MB/s)".format(baseline_label, baseline_mbps))
    for bar, val in zip(bars, values):
        if pd.isna(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                "{:.1f}x".format(val), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Speedup (MB/s ratio)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def plot_speedup_panels(df, baselines, output_path):
    df = df.copy()
    df["label"] = df.apply(
        lambda r: "{} t={}".format(r["kernel"].upper(), int(r["threads_per_block"])),
        axis=1,
    )
    df = df.sort_values(["kernel", "threads_per_block"]).reset_index(drop=True)
    colors = ["#3498db" if k == "v1" else "#e74c3c" for k in df["kernel"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    per_piece_mbps = baselines["per_piece_loop"]["mbps"]
    encode_mbps = baselines["encode_full"]["mbps"]

    _bar_panel(ax1, df, df["kernel_mbps"] / per_piece_mbps,
               per_piece_mbps, "tiktoken per-piece loop",
               "GPU kernel-only vs tiktoken per-piece loop\n(BPE merge algorithm only - excludes regex on both sides)",
               colors)

    if "e2e_mbps" in df.columns and df["e2e_mbps"].notna().any():
        _bar_panel(ax2, df, df["e2e_mbps"] / encode_mbps,
                   encode_mbps, "tiktoken encode()",
                   "GPU end-to-end vs tiktoken encode()\n(GPU side excludes Python regex pre-split)",
                   colors)
    else:
        ax2.text(0.5, 0.5, "No e2e timing in CSV", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11)
        ax2.set_axis_off()

    if "pipeline_mbps" in df.columns and df["pipeline_mbps"].notna().any():
        _bar_panel(ax3, df, df["pipeline_mbps"] / encode_mbps,
                   encode_mbps, "tiktoken encode()",
                   "GPU pipeline (e2e + Python regex) vs tiktoken encode()\n(apples-to-apples: both include regex + BPE + transfers)",
                   colors)
    else:
        ax3.text(0.5, 0.5, "No pipeline timing\n(needs e2e timing + regex baseline)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=11)
        ax3.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(output_path))


def write_report(df, baselines, output_path):
    lines = []
    lines.append("BPE Benchmark Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Tiktoken baselines (Python, measured on this run):")
    for name in ("encode_full", "per_piece_loop", "regex_only"):
        b = baselines.get(name)
        if not b:
            continue
        if "tokens_per_sec" in b:
            lines.append("  {:<16s}: {:7.3f} ms  ->  {:6.2f} MB/s  ({:.0f} tok/s)".format(
                name, b["avg_time_ms"], b["mbps"], b["tokens_per_sec"]))
        else:
            lines.append("  {:<16s}: {:7.3f} ms  ->  {:6.2f} MB/s".format(
                name, b["avg_time_ms"], b["mbps"]))
    lines.append("")

    per_piece_mbps = baselines["per_piece_loop"]["mbps"]
    encode_mbps = baselines["encode_full"]["mbps"]
    regex_ms = baselines["regex_only"]["avg_time_ms"]

    def best(df, col):
        if col not in df.columns or df[col].isna().all():
            return None
        return df.nlargest(1, col).iloc[0]

    for kernel in ("v1", "v2"):
        sub = df[df["kernel"] == kernel]
        if sub.empty:
            continue
        kbest = best(sub, "kernel_mbps")
        ebest = best(sub, "e2e_mbps")
        pbest = best(sub, "pipeline_mbps")
        lines.append("=== Best {} configurations ===".format(kernel.upper()))
        if kbest is not None:
            lines.append("Kernel-only (BPE merge work, excludes regex on both sides):")
            lines.append("  threads={} blocks={}  time={:.3f} ms  mbps={:.2f}  tok/s={:.0f}".format(
                int(kbest["threads_per_block"]), int(kbest["blocks"]),
                kbest["kernel_time_ms"], kbest["kernel_mbps"], kbest["kernel_tokens_per_sec"]))
            lines.append("  speedup vs tiktoken per_piece_loop: {:.2f}x".format(
                kbest["kernel_mbps"] / per_piece_mbps))
        if ebest is not None and not pd.isna(ebest["e2e_mbps"]):
            lines.append("End-to-end (kernel + transfers, GPU regex amortized off-clock):")
            lines.append("  threads={} blocks={}  time={:.3f} ms  mbps={:.2f}  tok/s={:.0f}".format(
                int(ebest["threads_per_block"]), int(ebest["blocks"]),
                ebest["e2e_time_ms"], ebest["e2e_mbps"], ebest["e2e_tokens_per_sec"]))
            lines.append("  speedup vs tiktoken encode():       {:.2f}x  (favors GPU - regex excluded)".format(
                ebest["e2e_mbps"] / encode_mbps))
        if pbest is not None and not pd.isna(pbest["pipeline_mbps"]):
            lines.append("Pipeline (apples-to-apples: e2e + Python regex pre-split):")
            lines.append("  threads={} blocks={}  time={:.3f} ms  ({:.3f} regex + {:.3f} e2e)".format(
                int(pbest["threads_per_block"]), int(pbest["blocks"]),
                pbest["pipeline_time_ms"], regex_ms, pbest["e2e_time_ms"]))
            lines.append("  mbps={:.2f}  tok/s={:.0f}".format(
                pbest["pipeline_mbps"], pbest["pipeline_tokens_per_sec"]))
            lines.append("  speedup vs tiktoken encode():       {:.2f}x  <-- truly fair number".format(
                pbest["pipeline_mbps"] / encode_mbps))
        lines.append("")

    lines.append("How to read these speedups:")
    lines.append("  - kernel-only vs per_piece_loop: how much faster the GPU's BPE merge")
    lines.append("    code is than tiktoken's Rust BPE core (both fed pre-split pieces).")
    lines.append("  - end-to-end vs encode(): how much faster a tokenizer service backed")
    lines.append("    by this kernel is, ASSUMING the regex pre-split is amortized")
    lines.append("    (cached, batched, or moved off the critical path).")
    lines.append("  - pipeline vs encode(): the truly apples-to-apples number. Includes")
    lines.append("    Python regex on the GPU side, matching tiktoken.encode()'s internal")
    lines.append("    Rust regex + BPE. Smaller speedup because Python regex is now the")
    lines.append("    new bottleneck - replacing it with PCRE2 in C++ would unlock the")
    lines.append("    full kernel-only speedup.")
    lines.append("")
    lines.append("All configurations:")
    lines.append(df.to_string(index=False))
    lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print("Saved {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description="Visualize BPE sweep CSV vs tiktoken")
    parser.add_argument("--csv", default="data/bpe_benchmark.csv")
    parser.add_argument("--pieces", default="data/pieces.bin")
    parser.add_argument("--ranks", default="data/bpe_ranks.bin")
    parser.add_argument("--text", required=True,
                        help="Original text file (used for the encode() baseline and regex timing)")
    parser.add_argument("--executable", default="./vocab_lookup.exe")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--iters", type=int, default=100,
                        help="Iterations for both the GPU sweep and the tiktoken baselines")
    parser.add_argument("--no-run", action="store_true",
                        help="Do not invoke the executable even if CSV is missing")
    args = parser.parse_args()

    if not args.no_run:
        maybe_run_sweep(args.executable, args.csv, args.iters, args.ranks, args.pieces)

    if not Path(args.csv).exists():
        raise SystemExit("CSV not found: {} (re-run without --no-run, or create it manually)".format(args.csv))

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty: {}".format(args.csv))
    df = normalize_columns(df)

    print("\nLoaded {} sweep rows from {}".format(len(df), args.csv))
    print(df.to_string(index=False))

    text = Path(args.text).read_text(encoding="utf-8")
    pieces, total_bytes = load_pieces(args.pieces)
    print("\nMeasuring tiktoken baselines ({} iterations)...".format(args.iters))
    baselines = measure_baselines(text, pieces, total_bytes, args.encoding, args.iters)
    for name, b in baselines.items():
        if "tokens_per_sec" in b:
            print("  {:<16s}: {:7.3f} ms  ->  {:6.2f} MB/s  ({:.0f} tok/s)".format(
                name, b["avg_time_ms"], b["mbps"], b["tokens_per_sec"]))
        else:
            print("  {:<16s}: {:7.3f} ms  ->  {:6.2f} MB/s".format(
                name, b["avg_time_ms"], b["mbps"]))

    df = add_pipeline_columns(df, baselines)

    plot_throughput_vs_threads(df, baselines, "throughput_vs_threads.png")
    plot_speedup_panels(df, baselines, "speedup_vs_tiktoken.png")
    write_report(df, baselines, "bpe_report.txt")


if __name__ == "__main__":
    main()
