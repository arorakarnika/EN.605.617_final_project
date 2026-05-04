#!/usr/bin/env bash
# Full scaling benchmark: fetch corpus if missing, export pieces per size,
# CUDA sweep with CSV tags, verify, combined scaling report.

ITERS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXE="${REPO_ROOT}/bpe_tokenizer.exe"
CSV_OUT="data/bpe_benchmark_scaling.csv"

UV="uv run python"

if [[ ! -f "${EXE}" ]]; then
  echo "ERROR: ${EXE} not found. Run 'make' in ${REPO_ROOT}" >&2
  exit 1
fi

mkdir -p data

if [[ ! -f data/corpus.txt ]]; then
  echo "=== Fetching combined corpus (all Gutenberg books in BOOKS) ==="
  ${UV} src/fetch_corpus.py --output data/corpus.txt
else
  echo "=== Using existing data/corpus.txt (delete to re-fetch combined corpus) ==="
fi

rm -f "${CSV_OUT}"

SIZES=(
  "corpus_1MB:data/corpus_1MB.txt"
  "corpus_5MB:data/corpus_5MB.txt"
  "corpus_10MB:data/corpus_10MB.txt"
  "corpus_20MB:data/corpus_20MB.txt"
  "corpus_full:data/corpus.txt"
)

FIRST=1
for entry in "${SIZES[@]}"; do
  TAG="${entry%%:*}"
  TEXT="${entry#*:}"
  if [[ ! -f "${TEXT}" ]]; then
    echo "ERROR: missing ${TEXT} (fetch_corpus should have created it)" >&2
    exit 1
  fi

  echo ""
  echo "========================================"
  echo "  Size tag: ${TAG}"
  echo "  Text:     ${TEXT}"
  echo "========================================"

  if [[ "${FIRST}" -eq 1 ]]; then
    echo "=== Export ranks + pieces ==="
    ${UV} src/export_tiktoken_vocab.py --text "${TEXT}"
    FIRST=0
  else
    echo "=== Export pieces (--skip-ranks) ==="
    ${UV} src/export_tiktoken_vocab.py --text "${TEXT}" --skip-ranks
  fi

  echo "=== CUDA sweep (tag=${TAG}) ==="
  "${EXE}" --sweep --csv "${CSV_OUT}" --tag "${TAG}" --iters "${ITERS}"

  echo "=== verify_bpe.py ==="
  ${UV} src/verify_bpe.py --text "${TEXT}"
done

echo ""
echo "=== bpe_scaling_report.txt + scaling PNGs ==="
${UV} src/bpe_visualizer.py --csv "${CSV_OUT}" --iters "${ITERS}"

echo ""
echo "Done. Master CSV: ${CSV_OUT}"
echo "Combined report: bpe_scaling_report.txt"
echo "Scaling plots: scaling_throughput_vs_size.png scaling_speedup_vs_size.png"
