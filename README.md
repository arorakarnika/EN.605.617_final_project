# GPU-Accelerated BPE Tokenization vs tiktoken

This project contains a CUDA implementation of OpenAI tiktoken's BPE merge algorithm, with end-to-end correctness verification and timing comparisons against the real `tiktoken` Python library.

The CUDA kernel produces token IDs that are **bit-for-bit identical** to `tiktoken.encode()` on the `cl100k_base` encoding (verified per piece by `src/verify_bpe.py`). This helps us make sure that the results being compared are in fact the same.

## Project Structure

```
EN.605.617_final_project/
├── src/
│   ├── main.cu                      # CLI parsing, sweep loop, dispatch
│   ├── bpe_kernels.cu               # __device__ helpers + V1 + V2 kernels
│   ├── bpe_io.cu                    # ranks/pieces loaders + CSV/token writers
│   ├── bpe_benchmark.cu             # run_bpe_benchmark host orchestration
│   ├── export_tiktoken_vocab.py     # exports tiktoken ranks + regex-pre-splits
│   ├── fetch_corpus.py              # downloads Gutenberg text(s), optional combine-all
│   ├── verify_bpe.py                # compares GPU output to tiktoken per piece
│   └── bpe_visualizer.py            # tagged scaling CSV, report + scaling PNGs
├── include/
│   └── bpe.h                        # shared structs and declarations
├── data/                            # generated artifacts (gitignored)
│   ├── bpe_ranks.bin                # tiktoken ranks in fixed-stride binary
│   ├── pieces.bin                   # text pre-split by tiktoken's regex
│   ├── corpus*.txt                  # downloaded text + truncated copies
│   ├── gpu_tokens.bin               # GPU output, consumed by verify_bpe.py
│   └── bpe_benchmark_scaling.csv    # optional: multi-size tagged sweep for visualizer
├── scaling_throughput_vs_size.png   # optional: from bpe_visualizer.py
├── scaling_speedup_vs_size.png
├── bpe_scaling_report.txt
├── scripts/
│   └── run_scaling_benchmark.sh     # full sweep across corpus sizes (tagged CSV)
├── Makefile
├── pyproject.toml                   # uv-managed Python deps
└── uv.lock
```


## Architecture

This project implements a tokenization pipeline that splits work between some python scripts and the CUDA kernels. The concerns are split as follows:
- Python: Download test text from Project Gutenberg, export tiktoken vocabulary and ranks, create pieces using tiktoken's regex pattern to ensure a fair comparison
- CUDA:
    - V1 Kernel: One thread per piece (each thread runs the whole sequential BPE merge loop in local memory)
    - V2 Kernel: One block per piece, all threads in the block cooperate on parallel pair-scoring + a 64-bit packed `(rank, position)` reduction in shared memory

## Setup

### CUDA

- NVIDIA CUDA Toolkit 11.0+
- GPU with compute capability `sm_75` (default in `Makefile`; may need to edit for your GPU)

```bash
make
```

Produces `bpe_tokenizer.exe` at the project root.

### Python (managed with `uv`)

If you don't have `uv` installed, you can install it using curl: 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See OS specific instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

## Full pipeline (end-to-end)

All commands are run from the project root.

### Fetch a corpus

Default (no `--book`): downloads **every** title in `BOOKS` inside `src/fetch_corpus.py`, strips Project Gutenberg boilerplate, concatenates with `=== book_key ===` markers between books, writes `data/corpus.txt`, then writes truncations at **1, 5, 10, and 50 MiB** (`data/corpus_1MB.txt` through `data/corpus_full.txt`) which is aroung 31mb with all books. Use `--sizes` to override (comma-separated byte counts).

```bash
uv run python src/fetch_corpus.py
```

Single book only (smaller download):

```bash
uv run python src/fetch_corpus.py --book complete_works_of_shakespeare
```

### Export tiktoken ranks and pre-split a corpus into pieces

```bash
uv run python src/export_tiktoken_vocab.py --text data/corpus.txt
```

Outputs:
- `data/bpe_ranks.bin` - 100,256 sorted-by-bytes BPE merge ranks (~13.6 MB).
- `data/pieces.bin` - text pre-split using tiktoken's exact regex pattern.

The ranks file only depends on the encoding, so for subsequent runs at different sizes pass `--skip-ranks`:

```bash
uv run python src/export_tiktoken_vocab.py --text data/corpus_5MB.txt --skip-ranks
```

### Run the GPU BPE encoder

Single config:

```bash
./bpe_tokenizer.exe                          # default: V2 kernel, 256 threads/block, 10 iters
./bpe_tokenizer.exe --kernel v1 --threads 256 --iters 100
```

Sweep V1 (32..1024 threads/block) + V2 (32..256 threads/block) and capture every config in one CSV:

```bash
./bpe_tokenizer.exe --sweep --csv data/bpe_benchmark_scaling.csv --tag corpus_1MB --iters 100
```

Available arguments

| Flag | Default | Meaning |
|------|---------|---------|
| `--kernel v1\|v2` | `v2` | Which kernel to launch |
| `--threads N` | `256` | Threads per block |
| `--blocks N` | auto | Grid size (V1 only; V2 always launches one block per piece) |
| `--iters N` | `10` | Timed iterations (warmup excluded from both timing passes) |
| `--ranks PATH` | `data/bpe_ranks.bin` | BPE ranks file |
| `--pieces PATH` | `data/pieces.bin` | Pre-split pieces file |
| `--output PATH` | `data/gpu_tokens.bin` | Where to write token IDs (empty string disables) |
| `--csv PATH` | (off) | Append one benchmark row per run; header is written when the file is empty |
| `--tag LABEL` | (empty) | Free-form label written into the CSV row, useful for grouping multiple sweeps |
| `--sweep` | (off) | Skip the single run; iterate V1+V2 thread counts and write one CSV row each |

### Verify GPU output matches tiktoken output

```bash
uv run python src/verify_bpe.py --text data/corpus.txt
```

Re-applies tiktoken's regex on the same text, runs `enc._encode_single_piece` on every piece, and compares to the GPU's IDs.

### Scaling report and plots

`src/bpe_visualizer.py` expects a CSV with a non-empty `tag` column (as produced by `scripts/run_scaling_benchmark.sh`). It measures tiktoken once per tag, writes `bpe_scaling_report.txt`, and writes `scaling_throughput_vs_size.png` and `scaling_speedup_vs_size.png` when matching `data/<tag>.txt` files exist.

```bash
uv run python src/bpe_visualizer.py --csv data/bpe_benchmark_scaling.csv --iters 100
```

For a full tagged sweep plus report, use `scripts/run_scaling_benchmark.sh` (see below).

### Full scaled benchmark

```bash
scripts/run_scaling_benchmark.sh
```

Runs the combined fetch if `data/corpus.txt` is missing, then sweeps at **1, 5, 10, 50 MiB** and **full** text (`tags` `corpus_1MB` ... `corpus_full`), appends `data/bpe_benchmark_scaling.csv`, verifies each size, and runs `src/bpe_visualizer.py` to write **`bpe_scaling_report.txt`** plus **`scaling_throughput_vs_size.png`** and **`scaling_speedup_vs_size.png`** when corpus files are present.

The report has one tiktoken baseline block per tag, a full GPU table per tag with `x_k_pp`, `x_e_enc`, `x_p_enc`, and the PNGs plot best MB/s vs input size (MiB) across tags.


Outputs:
- `bpe_scaling_report.txt`
- `scaling_throughput_vs_size.png`
- `scaling_speedup_vs_size.png`

### Understanding the Report

Baseline summary (under each `[tag]` heading):

| Column | Meaning |
|--------|--------|
| **encode_full** | One `tiktoken` `encode_ordinary(text)` call: Rust regex split + BPE. Time (ms) and throughput (MB/s) for the full input bytes. |
| **per_piece** | Python loop over pieces calling `_encode_single_piece` each time. Same BPE work as the GPU kernel input, time and MB/s. |
| **regex** | Python `regex.finditer` only (splitting cost). Used with GPU e2e to build **pipeline** time. |

GPU table columns:

| Column | Meaning |
|--------|--------|
| **kernel** | `v1` or `v2` CUDA implementation. |
| **threads_per_block** | CUDA block size for that run. |
| **blocks** | Grid size (V1: pieces mapped to grid; V2: one block per piece, so blocks = num_pieces). |
| **iterations** | Number of timed repetitions averaged for the reported times. |
| **num_pieces** | Count of regex pieces fed to the GPU for that corpus. |
| **input_bytes** | Total bytes in those pieces (same basis as MB/s). |
| **num_tokens** | Total BPE token IDs produced for that run. |
| **kernel_time_ms** | Mean time (ms) per iteration: device-resident pieces and ranks, kernel only, token output kept on device. |
| **x_k_pp** | Ratio of GPU **kernel** MB/s to tiktoken **per_piece** MB/s for this tag. MB/s = `(input_bytes / 1048576) / (time_ms / 1000)`. |
| **e2e_time_ms** | Mean time (ms) per iteration: H2D pieces + kernel + D2H tokens (ranks not re-uploaded each iteration). |
| **x_e_enc** | Ratio of GPU **e2e** MB/s to tiktoken **encode_full** MB/s for this tag. |
| **pipeline_time_ms** | `e2e_time_ms` + **regex** baseline (ms) for this tag: GPU path plus Python split to mirror `encode()` work split across host and device. |
| **x_p_enc** | Ratio of **pipeline** MB/s to tiktoken **encode_full** MB/s for this tag. |


## References

- OpenAI tiktoken: https://github.com/openai/tiktoken
- BPE algorithm: Sennrich et al. (2016), Neural Machine Translation of Rare Words with Subword Units
- Project Gutenberg: source for the benchmarking corpus (public domain)
