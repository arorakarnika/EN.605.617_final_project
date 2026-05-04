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
│   ├── fetch_corpus.py              # downloads a Project Gutenberg book
│   ├── verify_bpe.py                # compares GPU output to tiktoken per piece
│   └── bpe_visualizer.py            # reads CSV, measures baselines, makes plots
├── include/
│   └── bpe.h                        # shared structs and declarations
├── data/                            # generated artifacts (gitignored)
│   ├── bpe_ranks.bin                # tiktoken ranks in fixed-stride binary
│   ├── pieces.bin                   # text pre-split by tiktoken's regex
│   ├── corpus*.txt                  # downloaded text + truncated copies
│   ├── gpu_tokens.bin               # GPU output, consumed by verify_bpe.py
│   └── bpe_benchmark.csv            # checked in: sweep results used for visualiation
├── throughput_vs_threads.png        # checked in: sweep visualisation
├── speedup_vs_tiktoken.png          # checked in: 3-panel speedup chart
├── bpe_report.txt                   # checked in: text summary report
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

### 1. Fetch a corpus

```bash
uv run python src/fetch_corpus.py --book pride_and_prejudice
```

Writes `data/corpus.txt` plus `data/corpus_{1KB,10KB,100KB,1MB}.txt`. Other books available: `moby_dick`, `alice`, `frankenstein`, `tale_of_two_cities`.

### 2. Export tiktoken ranks and pre-split a corpus into pieces

```bash
uv run python src/export_tiktoken_vocab.py --text data/corpus.txt
```

Outputs:
- `data/bpe_ranks.bin` - 100,256 sorted-by-bytes BPE merge ranks (~13.6 MB).
- `data/pieces.bin` - text pre-split using tiktoken's exact regex pattern.

The ranks file only depends on the encoding, so for subsequent runs at different sizes pass `--skip-ranks`:

```bash
uv run python src/export_tiktoken_vocab.py --text data/corpus.txt
```

### 3. Run the GPU BPE encoder

Single config:

```bash
./bpe_tokenizer.exe                          # default: V2 kernel, 256 threads/block, 10 iters
./bpe_tokenizer.exe --kernel v1 --threads 256 --iters 100
```

Sweep V1 (32..1024 threads/block) + V2 (32..256 threads/block) and capture every config in one CSV:

```bash
./bpe_tokenizer.exe --sweep --csv data/bpe_benchmark.csv --iters 100
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

### 4. Verify GPU output matches tiktoken bit-for-bit

```bash
uv run python src/verify_bpe.py --text data/corpus.txt
```

Re-applies tiktoken's regex on the same text, runs `enc._encode_single_piece` on every piece, and compares to the GPU's IDs. Exits 0 with `PASS` when every piece matches.

### 5. Generate plots and report

```bash
uv run python src/bpe_visualizer.py \
    --csv data/bpe_benchmark.csv \
    --pieces data/pieces.bin \
    --text data/corpus.txt \
    --iters 100
```

Measures three Python baselines on the same corpus:

| Baseline | What it measures |
|---|---|
| `encode_full` | `enc.encode(text)` - one Rust call covering regex pre-split + per-piece BPE. Realistic production-tiktoken cost. |
| `per_piece_loop` | A Python loop calling `enc._encode_single_piece(piece)` per piece. Same BPE work as the GPU, but pays Python<->Rust boundary overhead per piece. |
| `regex_only` | Just `regex.finditer(text)` time. Combined with `e2e_time_ms` to derive the apples-to-apples pipeline number. |

We use this to derive the **truly fair** pipeline number for each GPU config (this is then used in the python visualizations)
```
pipeline_time_ms = e2e_time_ms + regex_only.avg_time_ms
```
This includes everything `tiktoken.encode()` does (regex split + per-piece BPE + transfers).

Outputs:
- `throughput_vs_threads.png` - per kernel: kernel-only (solid), end-to-end (dashed), pipeline including regex (dotted), with both tiktoken baselines as horizontal references.
- `speedup_vs_tiktoken.png` - three panels:
  1. GPU kernel-only vs `per_piece_loop` - "BPE merge algorithm" speedup (regex excluded on both sides).
  2. GPU end-to-end vs `encode()` - service throughput when regex is amortized off the critical path.
  3. GPU pipeline (e2e + Python regex) vs `encode()` - **the most fair comparison**.
- `bpe_report.txt` - text summary with best V1/V2 for kernel/e2e/pipeline

### 6. Full scaling sweep (all corpus sizes, tagged CSV)

From the repo root, after `make` and `uv sync`:

```bash
./scripts/run_scaling_benchmark.sh
```

This fetches the default Gutenberg book if `data/corpus.txt` is missing, deletes any previous `data/bpe_benchmark_scaling.csv`, then for each of `corpus_1KB`, `corpus_10KB`, `corpus_100KB`, `corpus_1MB`, and `corpus_full` (`data/corpus.txt`): exports `pieces.bin`, runs `./bpe_tokenizer.exe --sweep` with `--tag` set to that label, runs `verify_bpe.py`, and writes per-size plots under `data/scaling_plots/<tag>/`. Edit `ITERS` at the top of the script to change the timed iteration count. Delete `data/corpus.txt` before running if you want a fresh download. Use the `tag` column in the master CSV to compare throughput and speedup across input size.

## References

- OpenAI tiktoken: https://github.com/openai/tiktoken
- BPE algorithm: Sennrich et al. (2016), Neural Machine Translation of Rare Words with Subword Units
- Project Gutenberg: source for the benchmarking corpus (public domain)
