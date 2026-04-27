#ifndef VOCAB_LOOKUP_H
#define VOCAB_LOOKUP_H

#include <cuda_runtime.h>

#define MAX_TOKEN_LENGTH 64
#define MAX_VOCAB_SIZE 50000
#define UNK_TOKEN_ID -1

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

struct BenchmarkResult {
    float cpu_time_ms;
    float gpu_time_ms;
    int num_tokens;
    float cpu_throughput;
    float gpu_throughput;
    bool correct;
};

void load_vocabulary(const char* vocab_path, Vocabulary* vocab);
void free_vocabulary(Vocabulary* vocab);
void sort_vocabulary(Vocabulary* vocab);

int cpu_vocab_lookup(const char* token, int token_length, const Vocabulary* vocab);
void cpu_batch_vocab_lookup(const char* text, const int* token_offsets, 
                            const int* token_lengths, int num_tokens,
                            const Vocabulary* vocab, int* token_ids);

__global__ void gpu_vocab_lookup_kernel(const char* text, const int* token_offsets,
                                        const int* token_lengths, int num_tokens,
                                        const VocabEntry* vocab, int vocab_size,
                                        int* token_ids);

void run_benchmark(const char* text, const int* token_offsets, const int* token_lengths,
                  int num_tokens, const Vocabulary* vocab, BenchmarkResult* result);

#endif
