// Host orchestration for the BPE benchmark.
//
// Allocates device buffers, copies the (large, reused) ranks table once,
// then runs two timed passes:
//   1. Kernel-only: warm buffers already on device, outputs stay on device.
//   2. End-to-end:  H2D pieces + kernel + D2H token IDs per iteration. The
//                   ranks table is intentionally NOT re-uploaded - in any
//                   realistic deployment that 13.6 MB table is loaded once
//                   and reused across many requests.

#include <stdio.h>
#include <stdlib.h>
#include "bpe.h"

static void compute_token_offsets(const Pieces* pieces, int* offsets) {
    int running = 0;
    for (int i = 0; i < pieces->num_pieces; i++) {
        offsets[i] = running;
        running += pieces->lengths[i];
    }
}

void run_bpe_benchmark(const Pieces* pieces,
                       const BPERank* h_ranks,
                       int num_ranks,
                       const BPERunConfig* config) {
    const int num_pieces = pieces->num_pieces;
    if (num_pieces == 0) {
        fprintf(stderr, "No pieces to encode\n");
        return;
    }

    int* h_token_offsets = (int*)malloc(num_pieces * sizeof(int));
    compute_token_offsets(pieces, h_token_offsets);
    const int max_total_tokens = pieces->total_bytes;

    unsigned char* d_blob = NULL;
    int* d_offsets = NULL;
    int* d_lengths = NULL;
    BPERank* d_ranks = NULL;
    int* d_token_ids = NULL;
    int* d_token_offsets = NULL;
    int* d_token_counts = NULL;

    CHECK_CUDA(cudaMalloc(&d_blob, pieces->total_bytes));
    CHECK_CUDA(cudaMalloc(&d_offsets, num_pieces * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_lengths, num_pieces * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ranks, num_ranks * sizeof(BPERank)));
    CHECK_CUDA(cudaMalloc(&d_token_ids, max_total_tokens * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_token_offsets, num_pieces * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_token_counts, num_pieces * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_blob, pieces->blob, pieces->total_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, pieces->offsets,
                          num_pieces * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lengths, pieces->lengths,
                          num_pieces * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ranks, h_ranks, num_ranks * sizeof(BPERank),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_offsets, h_token_offsets,
                          num_pieces * sizeof(int), cudaMemcpyHostToDevice));

    int threads = config->threads_per_block > 0 ? config->threads_per_block : 256;
    if (threads > MAX_BLOCK_SIZE) threads = MAX_BLOCK_SIZE;

    int blocks;
    if (config->kernel_version == 2) {
        blocks = num_pieces;
        if (config->blocks > 0) {
            fprintf(stderr,
                    "Note: --blocks is ignored for V2 (uses 1 block per piece)\n");
        }
    } else {
        if (config->blocks > 0) {
            blocks = config->blocks;
        } else {
            blocks = (num_pieces + threads - 1) / threads;
        }
    }

    int iterations = config->iterations > 0 ? config->iterations : 1;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    if (config->kernel_version == 2) {
        bpe_encode_kernel_v2<<<blocks, threads>>>(
            d_blob, d_offsets, d_lengths, num_pieces,
            d_ranks, num_ranks,
            d_token_ids, d_token_offsets, d_token_counts);
    } else {
        bpe_encode_kernel_v1<<<blocks, threads>>>(
            d_blob, d_offsets, d_lengths, num_pieces,
            d_ranks, num_ranks,
            d_token_ids, d_token_offsets, d_token_counts,
            BPE_MAX_TOKENS_PER_PIECE);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Pass 1: kernel-only timing.
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < iterations; it++) {
        if (config->kernel_version == 2) {
            bpe_encode_kernel_v2<<<blocks, threads>>>(
                d_blob, d_offsets, d_lengths, num_pieces,
                d_ranks, num_ranks,
                d_token_ids, d_token_offsets, d_token_counts);
        } else {
            bpe_encode_kernel_v1<<<blocks, threads>>>(
                d_blob, d_offsets, d_lengths, num_pieces,
                d_ranks, num_ranks,
                d_token_ids, d_token_offsets, d_token_counts,
                BPE_MAX_TOKENS_PER_PIECE);
        }
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float kernel_total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_total_ms, start, stop));

    int* h_token_counts = (int*)malloc(num_pieces * sizeof(int));
    int* h_token_ids = (int*)malloc(max_total_tokens * sizeof(int));

    // Pass 2: end-to-end per-call timing.
    cudaEvent_t e2e_start, e2e_stop;
    CHECK_CUDA(cudaEventCreate(&e2e_start));
    CHECK_CUDA(cudaEventCreate(&e2e_stop));
    CHECK_CUDA(cudaEventRecord(e2e_start));
    for (int it = 0; it < iterations; it++) {
        CHECK_CUDA(cudaMemcpyAsync(d_blob, pieces->blob, pieces->total_bytes,
                                   cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(d_offsets, pieces->offsets,
                                   num_pieces * sizeof(int),
                                   cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(d_lengths, pieces->lengths,
                                   num_pieces * sizeof(int),
                                   cudaMemcpyHostToDevice));
        if (config->kernel_version == 2) {
            bpe_encode_kernel_v2<<<blocks, threads>>>(
                d_blob, d_offsets, d_lengths, num_pieces,
                d_ranks, num_ranks,
                d_token_ids, d_token_offsets, d_token_counts);
        } else {
            bpe_encode_kernel_v1<<<blocks, threads>>>(
                d_blob, d_offsets, d_lengths, num_pieces,
                d_ranks, num_ranks,
                d_token_ids, d_token_offsets, d_token_counts,
                BPE_MAX_TOKENS_PER_PIECE);
        }
        CHECK_CUDA(cudaMemcpyAsync(h_token_counts, d_token_counts,
                                   num_pieces * sizeof(int),
                                   cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpyAsync(h_token_ids, d_token_ids,
                                   max_total_tokens * sizeof(int),
                                   cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(e2e_stop));
    CHECK_CUDA(cudaEventSynchronize(e2e_stop));

    float e2e_total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&e2e_total_ms, e2e_start, e2e_stop));

    int total_tokens = 0;
    for (int i = 0; i < num_pieces; i++) total_tokens += h_token_counts[i];

    float kernel_ms = kernel_total_ms / iterations;
    float e2e_ms = e2e_total_ms / iterations;
    float kernel_sec = kernel_ms / 1000.0f;
    float e2e_sec = e2e_ms / 1000.0f;
    float input_mb = pieces->total_bytes / (1024.0f * 1024.0f);

    float kernel_mbps = input_mb / kernel_sec;
    float kernel_tps = total_tokens / kernel_sec;
    float e2e_mbps = input_mb / e2e_sec;
    float e2e_tps = total_tokens / e2e_sec;

    printf("\nBPE kernel V%d  | grid=%d  block=%d  iters=%d\n",
           config->kernel_version, blocks, threads, iterations);
    printf("  Pieces:        %d\n", num_pieces);
    printf("  Input bytes:   %d\n", pieces->total_bytes);
    printf("  Tokens:        %d\n", total_tokens);
    printf("  Kernel only:   %.3f ms  (%.2f MB/s, %.0f tok/s)\n",
           kernel_ms, kernel_mbps, kernel_tps);
    printf("  End-to-end:    %.3f ms  (%.2f MB/s, %.0f tok/s)  [H2D+kernel+D2H per call]\n",
           e2e_ms, e2e_mbps, e2e_tps);

    if (config->output_path) {
        write_token_output(config->output_path, h_token_ids,
                           h_token_offsets, h_token_counts, num_pieces);
    }

    if (config->csv_path) {
        append_bpe_csv_row(config->csv_path,
                           config->tag,
                           config->kernel_version,
                           threads,
                           blocks,
                           iterations,
                           num_pieces,
                           pieces->total_bytes,
                           total_tokens,
                           kernel_ms,
                           kernel_mbps,
                           kernel_tps,
                           e2e_ms,
                           e2e_mbps,
                           e2e_tps);
    }

    free(h_token_counts);
    free(h_token_ids);
    free(h_token_offsets);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(e2e_start));
    CHECK_CUDA(cudaEventDestroy(e2e_stop));
    CHECK_CUDA(cudaFree(d_blob));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_lengths));
    CHECK_CUDA(cudaFree(d_ranks));
    CHECK_CUDA(cudaFree(d_token_ids));
    CHECK_CUDA(cudaFree(d_token_offsets));
    CHECK_CUDA(cudaFree(d_token_counts));
}
