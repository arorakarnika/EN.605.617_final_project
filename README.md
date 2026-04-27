# GPU-Accelerated Vocabulary Lookup for Tokenization

## Overview

This project implements a GPU-accelerated tokenization pipeline. The implementation aims to benchmark the performance of a GPU-accelerated pipeline against tiktoken, a popular tokenizer that does not have GPU acceleration. 


## Setup Instructions

### Requirements

- NVIDIA CUDA Toolkit
- C++ compiler
- Python ^3.10
- NVIDIA GPU with compute capability

### Compilation

```bash
make
```

This compiles the project using nvcc with optimization flags.

### Usage

1. Generate synthetic test data:
```bash
./vocab_lookup.exe --generate
```

This creates:
- `data/sample_vocab.csv` - 1000 token vocabulary
- `data/sample_text.txt` - 10,000 word sample text

2. Run benchmark:
```bash
./vocab_lookup.exe
```

This will:
- Load vocabulary and text
- Tokenize the input text
- Run CPU and GPU vocabulary lookup
- Compare results for correctness
- Output performance metrics and CSV results

### Output

```
GPU Vocabulary Lookup Benchmark
================================

Loaded 1000 tokens into vocabulary
Tokenized 10000 tokens from text
Running benchmark...

Results:
  Tokens processed: 10000
  CPU time: X.XX ms (XXXXX tokens/sec)
  GPU time: X.XX ms (XXXXX tokens/sec)
  Speedup: X.XXx
  Correctness: PASS

Benchmark results written to benchmark_results.csv
```
