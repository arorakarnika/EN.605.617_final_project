#ifndef VOCAB_LOOKUP_H
#define VOCAB_LOOKUP_H

#include <cuda_runtime.h>
#include <stddef.h>

#define MAX_TOKEN_LENGTH 64
#define MAX_VOCAB_SIZE 50000
#define UNK_TOKEN_ID -1
#define MAX_BLOCK_SIZE 1024

// BPE-specific constants. Must match the layout produced by
// export_tiktoken_vocab.py.
#define BPE_MAX_TOKEN_LEN 128
#define BPE_MAX_PIECE_BYTES 256
#define BPE_MAX_TOKENS_PER_PIECE 256

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

struct VocabEntry {
    char token[MAX_TOKEN_LENGTH];
    int token_id;
    int token_length;
};

struct Vocabulary {
    VocabEntry* entries;
    int size;
    bool is_sorted;
};

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

struct BenchmarkResult {
    float cpu_time_ms;
    float gpu_time_ms;
    int num_tokens;
    size_t text_bytes;
    float cpu_throughput_tokens_per_sec;
    float gpu_throughput_tokens_per_sec;
    float cpu_throughput_mbps;
    float gpu_throughput_mbps;
    float speedup;
    bool correct;
};

// Existing simple-vocabulary path (kept for backwards compatibility).
void load_vocabulary(const char* vocab_path, Vocabulary* vocab);
void free_vocabulary(Vocabulary* vocab);
void sort_vocabulary(Vocabulary* vocab);

void cpu_normalize(const char* input, char* output, size_t n);
void populate_ascii_map();

__global__ void gpu_normalize_v1(const char* input, char* output, size_t n);
__global__ void gpu_normalize_v2(const char* input, char* output, size_t n);

int cpu_vocab_lookup(const char* token, int token_length, const Vocabulary* vocab);
void cpu_batch_vocab_lookup(const char* text, const int* token_offsets,
                            const int* token_lengths, int num_tokens,
                            const Vocabulary* vocab, int* token_ids);

__global__ void gpu_vocab_lookup_kernel(const char* text, const int* token_offsets,
                                        const int* token_lengths, int num_tokens,
                                        const VocabEntry* vocab, int vocab_size,
                                        int* token_ids);

void run_benchmark(const char* text, const int* token_offsets, const int* token_lengths,
                   int num_tokens, size_t text_bytes, const Vocabulary* vocab,
                   BenchmarkResult* result, int num_iterations);

void run_tiktoken_style_benchmark(const char* text_path, const Vocabulary* vocab);

// BPE pipeline: vocabulary loading.
void load_bpe_ranks(const char* path, BPERank** ranks_out, int* num_ranks_out);
void free_bpe_ranks(BPERank* ranks);

// BPE pipeline: pre-split pieces loading.
void load_pieces(const char* path, Pieces* pieces);
void free_pieces(Pieces* pieces);

// BPE kernels.
//   V1: one thread per piece. Each thread runs its own sequential BPE loop.
//   V2: one block per piece. Threads cooperate via shared memory to scan pairs
//       and reduce to the lowest-rank merge each round.
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

// Host orchestration for the BPE benchmark. Runs the chosen kernel, captures
// timing, and optionally writes the resulting token IDs to a binary file.
struct BPERunConfig {
    int kernel_version;      // 1 or 2
    int threads_per_block;   // V1 only (V2 uses one block per piece)
    int blocks;              // 0 means auto
    int iterations;
    const char* output_path; // NULL to skip writing token-id output
    const char* csv_path;    // NULL to skip CSV row append
    const char* tag;         // optional run label written to CSV
};

void run_bpe_benchmark(const Pieces* pieces,
                       const BPERank* h_ranks,
                       int num_ranks,
                       const BPERunConfig* config,
                       BenchmarkResult* result);

#endif
