#!/usr/bin/env bash
# Full scaling benchmark: fetch corpus if missing, export pieces per size,
# CUDA sweep with CSV tags, verify, per-size plots.

ITERS=100

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
  echo "=== Fetching corpus (pride_and_prejudice) ==="
  ${UV} src/fetch_corpus.py --book pride_and_prejudice --output data/corpus.txt
else
  echo "=== Using existing data/corpus.txt ==="
fi

rm -f "${CSV_OUT}"

SIZES=(
  "corpus_1KB:data/corpus_1KB.txt"
  "corpus_10KB:data/corpus_10KB.txt"
  "corpus_100KB:data/corpus_100KB.txt"
  "corpus_1MB:data/corpus_1MB.txt"
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

  PLOT_DIR="data/scaling_plots/${TAG}"
  mkdir -p "${PLOT_DIR}"
  FILTERED="${PLOT_DIR}/sweep_${TAG}.csv"
  awk -F',' -v tag="${TAG}" 'NR==1 || $1==tag' "${CSV_OUT}" > "${FILTERED}"
  echo "=== bpe_visualizer.py -> ${PLOT_DIR}/ ==="
  (
    cd "${PLOT_DIR}"
    ${UV} "${REPO_ROOT}/src/bpe_visualizer.py" --csv "${FILTERED}" \
      --pieces "${REPO_ROOT}/data/pieces.bin" \
      --text "${REPO_ROOT}/${TEXT}" \
      --executable "${EXE}" \
      --no-run \
      --iters "${ITERS}"
  )
done

echo ""
echo "Done. Master CSV: ${CSV_OUT}"
echo "Per-size outputs: data/scaling_plots/<tag>/"
