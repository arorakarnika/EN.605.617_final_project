#ifndef BPE_H
#define BPE_H

#include <cuda_runtime.h>
#include <stddef.h>

// Must match the layout produced by src/export_tiktoken_vocab.py.
#define BPE_MAX_TOKEN_LEN 128
#define BPE_MAX_TOKENS_PER_PIECE 256

// Upper bound on threads per block we'll ever request from the CLI.
#define MAX_BLOCK_SIZE 1024

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// One mergeable token from the tiktoken vocabulary. The whole table is
// sorted lexicographically by `bytes` so we can binary-search for the rank
// of any byte sequence on the device. For tiktoken encodings the rank is
// also the final token id.
struct BPERank {
    unsigned char bytes[BPE_MAX_TOKEN_LEN];
    int length;
    int rank;
};

// Pre-split text pieces produced by tiktoken's regex on the host.
struct Pieces {
    unsigned char* blob;     // concatenated bytes
    int* offsets;            // start offset of piece i within blob
    int* lengths;            // byte length of piece i
    int num_pieces;
    int total_bytes;
};

// Knobs for one benchmark run.
struct BPERunConfig {
    int kernel_version;      // 1 or 2
    int threads_per_block;   // V1 only (V2 always uses one block per piece)
    int blocks;              // 0 means auto
    int iterations;
    const char* output_path; // NULL to skip writing token-id output
    const char* csv_path;    // NULL to skip CSV row append
    const char* tag;         // optional run label written to CSV
};

// ---------------------------------------------------------------------------
// I/O (src/bpe_io.cu)
// ---------------------------------------------------------------------------

void load_bpe_ranks(const char* path, BPERank** ranks_out, int* num_ranks_out);
void free_bpe_ranks(BPERank* ranks);

void load_pieces(const char* path, Pieces* pieces);
void free_pieces(Pieces* pieces);

void write_token_output(const char* path,
                        const int* token_ids,
                        const int* token_offsets,
                        const int* token_counts,
                        int num_pieces);

void append_bpe_csv_row(const char* path,
                        const char* tag,
                        int kernel_version,
                        int threads_per_block,
                        int blocks,
                        int iterations,
                        int num_pieces,
                        int input_bytes,
                        int num_tokens,
                        float kernel_time_ms,
                        float kernel_mbps,
                        float kernel_tokens_per_sec,
                        float e2e_time_ms,
                        float e2e_mbps,
                        float e2e_tokens_per_sec);

// ---------------------------------------------------------------------------
// Kernels (src/bpe_kernels.cu)
// ---------------------------------------------------------------------------
//   V1: one thread per piece. Each thread runs its own sequential BPE loop
//       using a per-thread workspace.
//   V2: one block per piece. Threads cooperate via shared memory: every
//       thread scores one adjacent pair, a parallel reduction picks the
//       lowest-rank pair, and thread 0 performs the merge.

__global__ void bpe_encode_kernel_v1(const unsigned char* blob,
                                     const int* piece_offsets,
                                     const int* piece_lengths,
                                     int num_pieces,
                                     const BPERank* ranks,
                                     int num_ranks,
                                     int* out_token_ids,
                                     int* out_token_offsets,
                                     int* out_token_counts,
                                     int max_tokens_per_piece);

__global__ void bpe_encode_kernel_v2(const unsigned char* blob,
                                     const int* piece_offsets,
                                     const int* piece_lengths,
                                     int num_pieces,
                                     const BPERank* ranks,
                                     int num_ranks,
                                     int* out_token_ids,
                                     int* out_token_offsets,
                                     int* out_token_counts);

// ---------------------------------------------------------------------------
// Benchmark host orchestration (src/bpe_benchmark.cu)
// ---------------------------------------------------------------------------

void run_bpe_benchmark(const Pieces* pieces,
                       const BPERank* h_ranks,
                       int num_ranks,
                       const BPERunConfig* config);

#endif
