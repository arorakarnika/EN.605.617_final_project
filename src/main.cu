// CLI entry point for the GPU BPE tokenizer.
//
// Usage examples:
//   ./bpe_tokenizer.exe                                  # default: V2 kernel, 256 threads, 10 iters
//   ./bpe_tokenizer.exe --kernel v1 --threads 256 --iters 100
//   ./bpe_tokenizer.exe --sweep --csv data/bpe_benchmark.csv --iters 100

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe.h"

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --ranks PATH        Binary ranks file (default: data/bpe_ranks.bin)\n");
    fprintf(stderr, "  --pieces PATH       Binary pieces file (default: data/pieces.bin)\n");
    fprintf(stderr, "  --output PATH       Write token IDs (default: data/gpu_tokens.bin; use '' to skip)\n");
    fprintf(stderr, "  --kernel v1|v2      Which kernel to run (default: v2)\n");
    fprintf(stderr, "  --threads N         Threads per block (default: 256)\n");
    fprintf(stderr, "  --blocks N          Blocks (V1 only; default: ceil(pieces/threads))\n");
    fprintf(stderr, "  --iters N           Timed iterations (default: 10)\n");
    fprintf(stderr, "  --csv PATH          Append a benchmark row to CSV (default: skip)\n");
    fprintf(stderr, "  --tag LABEL         Free-form label written into the CSV row\n");
    fprintf(stderr, "  --sweep             Skip single run; run V1 and V2 across multiple thread counts,\n");
    fprintf(stderr, "                      writing one CSV row per configuration\n");
}

static int parse_kernel_version(const char* s) {
    if (strcmp(s, "v1") == 0 || strcmp(s, "1") == 0) return 1;
    if (strcmp(s, "v2") == 0 || strcmp(s, "2") == 0) return 2;
    fprintf(stderr, "Unknown kernel version: %s (expected v1|v2)\n", s);
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    const char* ranks_path = "data/bpe_ranks.bin";
    const char* pieces_path = "data/pieces.bin";
    const char* output_path = "data/gpu_tokens.bin";
    const char* csv_path = NULL;
    const char* tag = NULL;
    int kernel_version = 2;
    int threads = 256;
    int blocks = 0;
    int iterations = 10;
    int sweep = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--ranks") == 0 && i + 1 < argc) {
            ranks_path = argv[++i];
        } else if (strcmp(argv[i], "--pieces") == 0 && i + 1 < argc) {
            pieces_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
            if (output_path[0] == '\0') output_path = NULL;
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernel_version = parse_kernel_version(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            blocks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--tag") == 0 && i + 1 < argc) {
            tag = argv[++i];
        } else if (strcmp(argv[i], "--sweep") == 0) {
            sweep = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    printf("GPU BPE Tokenization Benchmark\n");
    printf("================================\n");

    BPERank* ranks = NULL;
    int num_ranks = 0;
    load_bpe_ranks(ranks_path, &ranks, &num_ranks);

    Pieces pieces = {0};
    load_pieces(pieces_path, &pieces);

    if (sweep) {
        // Single CSV file accumulates one row per (kernel, threads) config.
        // Token output is written only on the first run so verify_bpe.py
        // still has something to compare against (it should match across
        // configs since the algorithm is deterministic).
        const int v1_threads[] = {32, 64, 128, 256, 512, 1024};
        const int v2_threads[] = {32, 64, 128, 256};
        const int n_v1 = sizeof(v1_threads) / sizeof(v1_threads[0]);
        const int n_v2 = sizeof(v2_threads) / sizeof(v2_threads[0]);

        printf("\n=== BPE Sweep ===\n");
        printf("V1 thread counts: ");
        for (int i = 0; i < n_v1; i++) printf("%d ", v1_threads[i]);
        printf("\nV2 thread counts: ");
        for (int i = 0; i < n_v2; i++) printf("%d ", v2_threads[i]);
        printf("\nIterations per config: %d\n", iterations);
        if (csv_path) printf("CSV output: %s\n", csv_path);
        printf("\n");

        int run_idx = 0;
        for (int i = 0; i < n_v1; i++) {
            BPERunConfig cfg;
            cfg.kernel_version = 1;
            cfg.threads_per_block = v1_threads[i];
            cfg.blocks = 0;
            cfg.iterations = iterations;
            cfg.output_path = (run_idx == 0) ? output_path : NULL;
            cfg.csv_path = csv_path;
            cfg.tag = tag;
            run_bpe_benchmark(&pieces, ranks, num_ranks, &cfg);
            run_idx++;
        }
        for (int i = 0; i < n_v2; i++) {
            BPERunConfig cfg;
            cfg.kernel_version = 2;
            cfg.threads_per_block = v2_threads[i];
            cfg.blocks = 0;
            cfg.iterations = iterations;
            cfg.output_path = NULL;
            cfg.csv_path = csv_path;
            cfg.tag = tag;
            run_bpe_benchmark(&pieces, ranks, num_ranks, &cfg);
            run_idx++;
        }
    } else {
        BPERunConfig cfg;
        cfg.kernel_version = kernel_version;
        cfg.threads_per_block = threads;
        cfg.blocks = blocks;
        cfg.iterations = iterations;
        cfg.output_path = output_path;
        cfg.csv_path = csv_path;
        cfg.tag = tag;
        run_bpe_benchmark(&pieces, ranks, num_ranks, &cfg);
    }

    free_pieces(&pieces);
    free_bpe_ranks(ranks);
    return 0;
}
