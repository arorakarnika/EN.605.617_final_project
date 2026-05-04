"""Scaling sweep CSV: tiktoken baselines per tag, text report, throughput/speedup PNGs."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import regex
import seaborn as sns
import tiktoken

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10

TAG_TEXT_PATH = {
    "corpus_1MB": "data/corpus_1MB.txt",
    "corpus_5MB": "data/corpus_5MB.txt",
    "corpus_10MB": "data/corpus_10MB.txt",
    "corpus_20MB": "data/corpus_20MB.txt",
    "corpus_full": "data/corpus.txt",
}


def resolve_corpus_text_path(tag):
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
    avg_regex = time_iterations(lambda: [m.group(0) for m in pat.finditer(text)], iterations)
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


def normalize_columns(df):
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
    preferred = ["corpus_1MB", "corpus_5MB", "corpus_10MB", "corpus_20MB", "corpus_full"]
    out = [t for t in preferred if t in u]
    out.extend(sorted(u - set(out)))
    return out


def gather_scaling_tag_runs(df, encoding, iters):
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
    lines = [
        "BPE sweep vs tiktoken (all rows)",
        "GPU MB/s = (input_bytes / 1048576) / (time_ms / 1000).",
        "x_k_pp = GPU kernel MB/s / tiktoken per_piece; x_e_enc = e2e / encode; x_p_enc = pipeline / encode.",
        "",
    ]
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
        b = item["baselines"]
        lines.append("[{}] {}".format(item["tag"], item["rel"]))
        lines.append(
            "  encode_full {:.3f} ms {:.2f} MB/s  |  per_piece {:.3f} ms {:.2f} MB/s  |  regex {:.3f} ms {:.2f} MB/s".format(
                b["encode_full"]["avg_time_ms"], b["encode_full"]["mbps"],
                b["per_piece_loop"]["avg_time_ms"], b["per_piece_loop"]["mbps"],
                b["regex_only"]["avg_time_ms"], b["regex_only"]["mbps"]))
        lines.append("")
        lines.append(item["sub"][cols].to_string(index=False))
        lines.append("")
    Path(out_path).write_text("\n".join(lines))


def _best_max(sub_df, col):
    if sub_df.empty or col not in sub_df.columns or sub_df[col].isna().all():
        return float("nan")
    return float(sub_df[col].max())


def plot_scaling_vs_input_size(ordered, throughput_path, speedup_path):
    runs = [x for x in ordered if x["kind"] == "ok"]
    if not runs:
        return False
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

    for ax_ in (ax1, ax2):
        ax_.legend(fontsize=7, loc="best")
        ax_.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(speedup_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main():
    p = argparse.ArgumentParser(description="Scaling sweep CSV: report + PNGs vs tiktoken")
    p.add_argument("--csv", default="data/bpe_benchmark_scaling.csv")
    p.add_argument("--scaling-report", default="bpe_scaling_report.txt")
    p.add_argument("--encoding", default="cl100k_base")
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    cpath = Path(args.csv)
    if not cpath.is_file():
        raise SystemExit("missing {}".format(args.csv))
    df = normalize_columns(pd.read_csv(cpath))
    if df.empty:
        raise SystemExit("empty CSV")
    tags = collect_tags(df)
    if not tags:
        raise SystemExit("CSV needs non-empty tag column")

    ordered = gather_scaling_tag_runs(df, args.encoding, args.iters)
    write_scaling_comparison_report(ordered, args.scaling_report)
    wrote_png = plot_scaling_vs_input_size(
        ordered, "scaling_throughput_vs_size.png", "scaling_speedup_vs_size.png")
    if wrote_png:
        print("wrote {} + scaling PNGs".format(args.scaling_report))
    else:
        print("wrote {} (no PNGs: missing corpus files for tags)".format(args.scaling_report))


if __name__ == "__main__":
    main()
