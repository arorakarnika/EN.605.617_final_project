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


def plot_throughput_vs_threads(df, baselines, output_path):
    fig, (ax_mbps, ax_tps) = plt.subplots(1, 2, figsize=(14, 5))

    style = {
        ("v1", "kernel"): {"color": "#3498db", "linestyle": "-",  "label": "GPU V1 kernel"},
        ("v1", "e2e"):    {"color": "#3498db", "linestyle": "--", "label": "GPU V1 end-to-end"},
        ("v2", "kernel"): {"color": "#e74c3c", "linestyle": "-",  "label": "GPU V2 kernel"},
        ("v2", "e2e"):    {"color": "#e74c3c", "linestyle": "--", "label": "GPU V2 end-to-end"},
    }

    for kernel in ("v1", "v2"):
        sub = df[df["kernel"] == kernel].sort_values("threads_per_block")
        if sub.empty:
            continue
        for which, mbps_col, tps_col in [
            ("kernel", "kernel_mbps", "kernel_tokens_per_sec"),
            ("e2e", "e2e_mbps", "e2e_tokens_per_sec"),
        ]:
            if sub[mbps_col].isna().all():
                continue
            s = style[(kernel, which)]
            ax_mbps.plot(sub["threads_per_block"], sub[mbps_col],
                         marker="o", linewidth=2, **s)
            ax_tps.plot(sub["threads_per_block"], sub[tps_col],
                        marker="o", linewidth=2, **s)

    for name, info in [
        ("encode_full",    {"color": "#555555", "linestyle": ":"}),
        ("per_piece_loop", {"color": "#888888", "linestyle": ":"}),
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
        ax.legend(fontsize=8)
    ax_mbps.set_ylabel("Throughput (MB/s)")
    ax_tps.set_ylabel("Throughput (tokens/sec)")
    ax_mbps.set_title("BPE encode throughput - MB/s")
    ax_tps.set_title("BPE encode throughput - tokens/sec")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(output_path))


def plot_speedup_panels(df, baselines, output_path):
    df = df.copy()
    df["label"] = df.apply(
        lambda r: "{} t={}".format(r["kernel"].upper(), int(r["threads_per_block"])),
        axis=1,
    )
    df = df.sort_values(["kernel", "threads_per_block"]).reset_index(drop=True)
    colors = ["#3498db" if k == "v1" else "#e74c3c" for k in df["kernel"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    per_piece_mbps = baselines["per_piece_loop"]["mbps"]
    speedup_kernel = df["kernel_mbps"] / per_piece_mbps
    bars = ax1.bar(df["label"], speedup_kernel, color=colors, alpha=0.85)
    ax1.axhline(1.0, color="gray", linestyle="--",
                label="tiktoken per-piece loop ({:.1f} MB/s)".format(per_piece_mbps))
    for bar, val in zip(bars, speedup_kernel):
        ax1.text(bar.get_x() + bar.get_width() / 2, val,
                 "{:.1f}x".format(val), ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Speedup (MB/s ratio)")
    ax1.set_title("GPU kernel-only vs tiktoken per-piece loop\n(BPE merge algorithm only)")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=30)
    for label in ax1.get_xticklabels():
        label.set_horizontalalignment("right")

    if df["e2e_mbps"].notna().any():
        encode_mbps = baselines["encode_full"]["mbps"]
        speedup_e2e = df["e2e_mbps"] / encode_mbps
        bars2 = ax2.bar(df["label"], speedup_e2e, color=colors, alpha=0.85)
        ax2.axhline(1.0, color="gray", linestyle="--",
                    label="tiktoken encode() ({:.1f} MB/s)".format(encode_mbps))
        for bar, val in zip(bars2, speedup_e2e):
            ax2.text(bar.get_x() + bar.get_width() / 2, val,
                     "{:.1f}x".format(val), ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("Speedup (MB/s ratio)")
        ax2.set_title("GPU end-to-end vs tiktoken encode()\n(realistic pipeline comparison)")
        ax2.legend(fontsize=8)
        ax2.tick_params(axis="x", rotation=30)
        for label in ax2.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        ax2.text(0.5, 0.5,
                 "No e2e timing in CSV\n(rerun the sweep with the new binary)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_axis_off()

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
        if kbest is not None:
            lines.append("Best {} kernel-only: threads={} blocks={}".format(
                kernel.upper(), int(kbest["threads_per_block"]), int(kbest["blocks"])))
            lines.append("  time={:.3f} ms  mbps={:.2f}  tok/s={:.0f}".format(
                kbest["kernel_time_ms"], kbest["kernel_mbps"], kbest["kernel_tokens_per_sec"]))
            lines.append("  speedup vs per-piece loop: {:.2f}x  (BPE-merge-only comparison)".format(
                kbest["kernel_mbps"] / per_piece_mbps))
        if ebest is not None and not pd.isna(ebest["e2e_mbps"]):
            lines.append("Best {} end-to-end: threads={} blocks={}".format(
                kernel.upper(), int(ebest["threads_per_block"]), int(ebest["blocks"])))
            lines.append("  time={:.3f} ms  mbps={:.2f}  tok/s={:.0f}".format(
                ebest["e2e_time_ms"], ebest["e2e_mbps"], ebest["e2e_tokens_per_sec"]))
            lines.append("  speedup vs encode():       {:.2f}x  (pipeline comparison)".format(
                ebest["e2e_mbps"] / encode_mbps))
        lines.append("")

    lines.append("Note: GPU end-to-end excludes the Python regex pre-split, which")
    lines.append("tiktoken.encode() does internally in Rust. Python regex_only timing")
    lines.append("above lets you mentally combine: realistic pipeline = e2e + regex_only.")
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

    plot_throughput_vs_threads(df, baselines, "throughput_vs_threads.png")
    plot_speedup_panels(df, baselines, "speedup_vs_tiktoken.png")
    write_report(df, baselines, "bpe_report.txt")


if __name__ == "__main__":
    main()
