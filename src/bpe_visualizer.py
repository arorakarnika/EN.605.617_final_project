"""
Visualize BPE benchmark results from the CUDA sweep CSV.
"""

import argparse
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

# Corpus tags from scripts/run_scaling_benchmark.sh -> text path for tiktoken baselines.
TAG_TEXT_PATH = {
    "corpus_1MB": "data/corpus_1MB.txt",
    "corpus_5MB": "data/corpus_5MB.txt",
    "corpus_10MB": "data/corpus_10MB.txt",
    "corpus_50MB": "data/corpus_50MB.txt",
    "corpus_full": "data/corpus.txt",
}


def resolve_corpus_text_path(tag):
    """Map CSV tag to UTF-8 text file for tiktoken baselines. Prefer explicit
    TAG_TEXT_PATH; otherwise use data/<tag>.txt when present (older sweeps)."""
    mapped = TAG_TEXT_PATH.get(tag)
    if mapped is not None:
        return mapped
    cand = Path("data") / "{}.txt".format(tag)
    if cand.is_file():
        return str(cand)
    return None


def pieces_from_text(text, encoding_name, max_piece_bytes=256):
    enc = tiktoken.get_encoding(encoding_name)
    pat = regex.compile(enc._pat_str)
    pieces = []
    for m in pat.finditer(text):
        pb = m.group(0).encode("utf-8")
        if not pb:
            continue
        if len(pb) > max_piece_bytes:
            continue
        pieces.append(pb)
    total = sum(len(p) for p in pieces)
    return pieces, total


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
    print("Running sweep: {} --sweep --csv {}".format(executable, csv_path))
    subprocess.run(
        [
            executable, "--sweep", "--csv", csv_path,
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


def collect_tags(df):
    if "tag" not in df.columns:
        return []
    s = df["tag"].fillna("").astype(str).str.strip()
    u = {x for x in s.unique().tolist() if x}
    preferred = ["corpus_1MB", "corpus_5MB", "corpus_10MB", "corpus_50MB", "corpus_full"]
    out = [t for t in preferred if t in u]
    out.extend(sorted(u - set(out)))
    return out


def gather_scaling_tag_runs(df, encoding, iters):
    """One tiktoken baseline pass per tag, in collect_tags order."""
    ordered = []
    for tag in collect_tags(df):
        rel = resolve_corpus_text_path(tag)
        if not rel:
            ordered.append({"kind": "err", "tag": tag, "msg": "no text path mapping"})
            continue
        tp = Path(rel)
        if not tp.is_file():
            ordered.append({"kind": "err", "tag": tag, "msg": "missing {}".format(rel)})
            continue
        text = tp.read_text(encoding="utf-8")
        pieces, tb = pieces_from_text(text, encoding)
        b = measure_baselines(text, pieces, tb, encoding, iters)
        sub = df[df["tag"].fillna("").astype(str).str.strip() == tag].copy()
        if sub.empty:
            ordered.append({"kind": "err", "tag": tag, "msg": "no CSV rows"})
            continue
        sub = add_pipeline_columns(sub, b)
        pp = b["per_piece_loop"]["mbps"]
        enc_mbps = b["encode_full"]["mbps"]
        sub = sub.copy()
        sub["x_k_pp"] = sub["kernel_mbps"] / pp
        sub["x_e_enc"] = sub["e2e_mbps"] / enc_mbps
        sub["x_p_enc"] = sub["pipeline_mbps"] / enc_mbps
        ordered.append({
            "kind": "ok",
            "tag": tag,
            "rel": rel,
            "baselines": b,
            "sub": sub,
        })
    return ordered


def write_scaling_comparison_report(ordered, out_path):
    lines = []
    lines.append("BPE sweep vs tiktoken (all rows)")
    lines.append("GPU MB/s = (input_bytes / 1048576) / (time_ms / 1000).")
    lines.append("x_k_pp = GPU kernel MB/s / tiktoken per_piece; x_e_enc = e2e / encode; x_p_enc = pipeline / encode.")
    lines.append("")
    cols = [
        "kernel", "threads_per_block", "blocks", "iterations",
        "num_pieces", "input_bytes", "num_tokens",
        "kernel_time_ms", "x_k_pp",
        "e2e_time_ms", "x_e_enc",
        "pipeline_time_ms", "x_p_enc",
    ]
    for item in ordered:
        if item["kind"] == "err":
            lines.append("[{}] {}".format(item["tag"], item["msg"]))
            lines.append("")
            continue
        tag = item["tag"]
        rel = item["rel"]
        b = item["baselines"]
        sub = item["sub"]
        lines.append("[{}] {}".format(tag, rel))
        lines.append(
            "  encode_full {:.3f} ms {:.2f} MB/s  |  per_piece {:.3f} ms {:.2f} MB/s  |  regex {:.3f} ms {:.2f} MB/s".format(
                b["encode_full"]["avg_time_ms"], b["encode_full"]["mbps"],
                b["per_piece_loop"]["avg_time_ms"], b["per_piece_loop"]["mbps"],
                b["regex_only"]["avg_time_ms"], b["regex_only"]["mbps"]))
        lines.append("")
        lines.append(sub[cols].to_string(index=False))
        lines.append("")
    Path(out_path).write_text("\n".join(lines))
    print("Saved {}".format(out_path))


def _best_max(sub_df, col):
    if sub_df.empty or col not in sub_df.columns or sub_df[col].isna().all():
        return float("nan")
    return float(sub_df[col].max())


def plot_scaling_vs_input_size(ordered, throughput_path, speedup_path):
    """Best GPU MB/s at each corpus size vs tiktoken (multi-tag CSV only)."""
    runs = [x for x in ordered if x["kind"] == "ok"]
    if not runs:
        print("Skipping scaling plots (no tags with corpus files and CSV rows).")
        return
    rows = []
    for item in runs:
        sub = item["sub"]
        b = item["baselines"]
        nbytes = int(sub["input_bytes"].iloc[0])
        mb = nbytes / (1024.0 * 1024.0)
        v1 = sub[sub["kernel"] == "v1"]
        v2 = sub[sub["kernel"] == "v2"]
        rows.append({
            "tag": item["tag"],
            "mb": mb,
            "encode_mbps": b["encode_full"]["mbps"],
            "per_piece_mbps": b["per_piece_loop"]["mbps"],
            "v1_kernel": _best_max(v1, "kernel_mbps"),
            "v2_kernel": _best_max(v2, "kernel_mbps"),
            "v1_pipe": _best_max(v1, "pipeline_mbps"),
            "v2_pipe": _best_max(v2, "pipeline_mbps"),
        })
    rows.sort(key=lambda r: r["mb"])
    mbs = [r["mb"] for r in rows]
    enc = [r["encode_mbps"] for r in rows]
    pp = [r["per_piece_mbps"] for r in rows]
    v1k = [r["v1_kernel"] for r in rows]
    v2k = [r["v2_kernel"] for r in rows]
    v1p = [r["v1_pipe"] for r in rows]
    v2p = [r["v2_pipe"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mbs, v1k, marker="o", color="#3498db", linewidth=2, label="GPU V1 kernel (best)")
    ax.plot(mbs, v2k, marker="o", color="#e74c3c", linewidth=2, label="GPU V2 kernel (best)")
    ax.plot(mbs, v1p, marker="s", color="#3498db", linestyle="--", linewidth=1.8, alpha=0.85,
            label="GPU V1 pipeline (best)")
    ax.plot(mbs, v2p, marker="s", color="#e74c3c", linestyle="--", linewidth=1.8, alpha=0.85,
            label="GPU V2 pipeline (best)")
    ax.plot(mbs, enc, marker="^", color="#222222", linestyle="-.", linewidth=1.5, label="tiktoken encode()")
    ax.plot(mbs, pp, marker="^", color="#777777", linestyle="-.", linewidth=1.5, label="tiktoken per-piece loop")
    ax.set_xlabel("Input size (MiB)")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Scaling: best sweep config per corpus tag")
    if len(mbs) >= 2 and max(mbs) / max(min(mbs), 1e-9) > 4:
        ax.set_xscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(throughput_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(throughput_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    enc_a = [e if e > 0 else float("nan") for e in enc]
    pp_a = [p if p > 0 else float("nan") for p in pp]
    ax1.plot(mbs, [v / e for v, e in zip(v1k, enc_a)], marker="o", color="#3498db", label="V1 kernel / encode")
    ax1.plot(mbs, [v / e for v, e in zip(v2k, enc_a)], marker="o", color="#e74c3c", label="V2 kernel / encode")
    ax1.plot(mbs, [v / e for v, e in zip(v1p, enc_a)], marker="s", color="#3498db", linestyle="--", alpha=0.85,
             label="V1 pipeline / encode")
    ax1.plot(mbs, [v / e for v, e in zip(v2p, enc_a)], marker="s", color="#e74c3c", linestyle="--", alpha=0.85,
             label="V2 pipeline / encode")
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax1.set_ylabel("Speedup vs tiktoken encode (MB/s ratio)")
    ax1.set_xlabel("Input size (MiB)")
    ax1.set_title("vs encode()")
    if len(mbs) >= 2 and max(mbs) / max(min(mbs), 1e-9) > 4:
        ax1.set_xscale("log")

    ax2.plot(mbs, [v / p for v, p in zip(v1k, pp_a)], marker="o", color="#3498db", label="V1 kernel / per-piece")
    ax2.plot(mbs, [v / p for v, p in zip(v2k, pp_a)], marker="o", color="#e74c3c", label="V2 kernel / per-piece")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Speedup vs tiktoken per-piece (MB/s ratio)")
    ax2.set_xlabel("Input size (MiB)")
    ax2.set_title("vs per-piece loop (BPE-only baseline)")
    if len(mbs) >= 2 and max(mbs) / max(min(mbs), 1e-9) > 4:
        ax2.set_xscale("log")

    for ax in (ax1, ax2):
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(speedup_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(speedup_path))


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
               "Kernel-only vs per-piece loop",
               colors)

    if "e2e_mbps" in df.columns and df["e2e_mbps"].notna().any():
        _bar_panel(ax2, df, df["e2e_mbps"] / encode_mbps,
                   encode_mbps, "tiktoken encode()",
                   "End-to-end vs encode()",
                   colors)
    else:
        ax2.text(0.5, 0.5, "No e2e timing in CSV", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11)
        ax2.set_axis_off()

    if "pipeline_mbps" in df.columns and df["pipeline_mbps"].notna().any():
        _bar_panel(ax3, df, df["pipeline_mbps"] / encode_mbps,
                   encode_mbps, "tiktoken encode()",
                   "Pipeline vs encode()",
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
    lines.append("BPE benchmark vs tiktoken")
    lines.append("")
    for name in ("encode_full", "per_piece_loop", "regex_only"):
        b = baselines.get(name)
        if not b:
            continue
        if "tokens_per_sec" in b:
            lines.append("  {:<16s}  {:7.3f} ms  {:6.2f} MB/s  {:.0f} tok/s".format(
                name, b["avg_time_ms"], b["mbps"], b["tokens_per_sec"]))
        else:
            lines.append("  {:<16s}  {:7.3f} ms  {:6.2f} MB/s".format(
                name, b["avg_time_ms"], b["mbps"]))
    lines.append("")
    lines.append("GPU MB/s = (input_bytes / 1048576) / (time_ms / 1000).")
    lines.append("x_k_pp = GPU kernel MB/s / tiktoken per_piece; x_e_enc = e2e / encode; x_p_enc = pipeline / encode.")
    lines.append("")

    per_piece_mbps = baselines["per_piece_loop"]["mbps"]
    encode_mbps = baselines["encode_full"]["mbps"]
    regex_ms = baselines["regex_only"]["avg_time_ms"]

    def best(df_, col):
        if col not in df_.columns or df_[col].isna().all():
            return None
        return df_.nlargest(1, col).iloc[0]

    for kernel in ("v1", "v2"):
        sub = df[df["kernel"] == kernel]
        if sub.empty:
            continue
        kbest = best(sub, "kernel_mbps")
        ebest = best(sub, "e2e_mbps")
        pbest = best(sub, "pipeline_mbps")
        lines.append("Best {}:".format(kernel.upper()))
        if kbest is not None:
            lines.append("  kernel  t={} blk={}  {:.3f} ms  {:.2f} MB/s  x_per_piece={:.2f}".format(
                int(kbest["threads_per_block"]), int(kbest["blocks"]),
                kbest["kernel_time_ms"], kbest["kernel_mbps"],
                kbest["kernel_mbps"] / per_piece_mbps))
        if ebest is not None and not pd.isna(ebest["e2e_mbps"]):
            lines.append("  e2e     t={} blk={}  {:.3f} ms  {:.2f} MB/s  x_encode={:.2f}".format(
                int(ebest["threads_per_block"]), int(ebest["blocks"]),
                ebest["e2e_time_ms"], ebest["e2e_mbps"],
                ebest["e2e_mbps"] / encode_mbps))
        if pbest is not None and not pd.isna(pbest["pipeline_mbps"]):
            lines.append("  pipeline t={} blk={}  {:.3f} ms  ({:.3f} regex + {:.3f} e2e)  {:.2f} MB/s  x_encode={:.2f}".format(
                int(pbest["threads_per_block"]), int(pbest["blocks"]),
                pbest["pipeline_time_ms"], regex_ms, pbest["e2e_time_ms"],
                pbest["pipeline_mbps"], pbest["pipeline_mbps"] / encode_mbps))
        lines.append("")

    df2 = df.copy()
    df2["x_k_pp"] = df2["kernel_mbps"] / per_piece_mbps
    df2["x_e_enc"] = df2["e2e_mbps"] / encode_mbps
    df2["x_p_enc"] = df2["pipeline_mbps"] / encode_mbps
    cols = [
        "kernel", "threads_per_block", "blocks", "iterations",
        "num_pieces", "input_bytes", "num_tokens",
        "kernel_time_ms", "x_k_pp",
        "e2e_time_ms", "x_e_enc",
        "pipeline_time_ms", "x_p_enc",
    ]
    if "tag" in df2.columns and df2["tag"].fillna("").astype(str).str.strip().ne("").any():
        cols = ["tag"] + cols
    lines.append("All runs")
    lines.append(df2[cols].to_string(index=False))

    Path(output_path).write_text("\n".join(lines))
    print("Saved {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description="Visualize BPE sweep CSV vs tiktoken")
    parser.add_argument("--csv", default="data/bpe_benchmark.csv")
    parser.add_argument("--pieces", default="data/pieces.bin")
    parser.add_argument("--ranks", default="data/bpe_ranks.bin")
    parser.add_argument("--text", default=None,
                        help="Text for baselines; optional if CSV has one known scaling tag")
    parser.add_argument("--scaling-report", default="bpe_scaling_report.txt",
                        help="Output path when CSV has multiple tags")
    parser.add_argument("--executable", default="./bpe_tokenizer.exe")
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

    tags = collect_tags(df)
    if len(tags) > 1:
        print("\nMulti-tag CSV; writing {} and scaling size plots.".format(args.scaling_report))
        ordered = gather_scaling_tag_runs(df, args.encoding, args.iters)
        write_scaling_comparison_report(ordered, args.scaling_report)
        plot_scaling_vs_input_size(
            ordered,
            "scaling_throughput_vs_size.png",
            "scaling_speedup_vs_size.png",
        )
        return

    text_path = args.text
    if text_path is None:
        if len(tags) == 1:
            cand = resolve_corpus_text_path(tags[0])
            if cand and Path(cand).is_file():
                text_path = cand
        if text_path is None:
            raise SystemExit(
                "Pass --text for baselines, or use a CSV with multiple tag values (writes {} and scaling PNGs).".format(
                    args.scaling_report))

    text = Path(text_path).read_text(encoding="utf-8")
    pieces, total_bytes = pieces_from_text(text, args.encoding)
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
