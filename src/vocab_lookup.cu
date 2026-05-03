#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <chrono>
#include <algorithm>
#include "vocab_lookup.h"

__constant__ char ascii_map[128];

void populate_ascii_map() {
    char h_map[128];
    for (int i = 0; i < 128; i++) {
        h_map[i] = (i >= 'A' && i <= 'Z') ? i + 32 : i;
    }
    CHECK_CUDA(cudaMemcpyToSymbol(ascii_map, h_map, 128 * sizeof(char)));
}

void cpu_normalize(const char* input, char* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)input[i];
        output[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
}

__global__ void gpu_normalize_v1(const char* input, char* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned char c = (unsigned char)input[idx];
        output[idx] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
}

__global__ void gpu_normalize_v2(const char* input, char* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned char c = (unsigned char)input[idx];
        output[idx] = (c < 128) ? ascii_map[c] : c;
    }
}

void load_vocabulary(const char* vocab_path, Vocabulary* vocab) {
    FILE* fp = fopen(vocab_path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open vocabulary file: %s\n", vocab_path);
        exit(EXIT_FAILURE);
    }
    
    vocab->entries = (VocabEntry*)malloc(MAX_VOCAB_SIZE * sizeof(VocabEntry));
    if (!vocab->entries) {
        fprintf(stderr, "Failed to allocate memory for vocabulary\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    
    vocab->size = 0;
    vocab->is_sorted = false;
    
    char line[256];
    while (fgets(line, sizeof(line), fp) && vocab->size < MAX_VOCAB_SIZE) {
        char token[MAX_TOKEN_LENGTH];
        int token_id;
        
        if (sscanf(line, "%d,%s", &token_id, token) == 2) {
            int len = strlen(token);
            if (len > 0 && token[len-1] == '\n') {
                token[len-1] = '\0';
                len--;
            }
            
            vocab->entries[vocab->size].token_id = token_id;
            vocab->entries[vocab->size].token_length = len;
            strncpy(vocab->entries[vocab->size].token, token, MAX_TOKEN_LENGTH - 1);
            vocab->entries[vocab->size].token[MAX_TOKEN_LENGTH - 1] = '\0';
            vocab->size++;
        }
    }
    
    fclose(fp);
    printf("Loaded %d tokens into vocabulary\n", vocab->size);
}

void free_vocabulary(Vocabulary* vocab) {
    if (vocab->entries) {
        free(vocab->entries);
        vocab->entries = NULL;
    }
    vocab->size = 0;
}

int compare_vocab_entries(const void* a, const void* b) {
    const VocabEntry* entry_a = (const VocabEntry*)a;
    const VocabEntry* entry_b = (const VocabEntry*)b;
    return strcmp(entry_a->token, entry_b->token);
}

void sort_vocabulary(Vocabulary* vocab) {
    if (!vocab->is_sorted) {
        qsort(vocab->entries, vocab->size, sizeof(VocabEntry), compare_vocab_entries);
        vocab->is_sorted = true;
    }
}

__device__ int strncmp_device(const char* s1, const char* s2, int n) {
    for (int i = 0; i < n; i++) {
        if (s1[i] != s2[i]) {
            return (unsigned char)s1[i] - (unsigned char)s2[i];
        }
        if (s1[i] == '\0') {
            return 0;
        }
    }
    return 0;
}

__device__ int binary_search_vocab(const char* token, int token_length,
                                   const VocabEntry* vocab, int vocab_size) {
    int left = 0;
    int right = vocab_size - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        int cmp = strncmp_device(token, vocab[mid].token, token_length);
        
        if (cmp == 0 && vocab[mid].token_length == token_length) {
            return vocab[mid].token_id;
        } else if (cmp < 0 || (cmp == 0 && token_length < vocab[mid].token_length)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return UNK_TOKEN_ID;
}

__global__ void gpu_vocab_lookup_kernel(const char* text, const int* token_offsets,
                                        const int* token_lengths, int num_tokens,
                                        const VocabEntry* vocab, int vocab_size,
                                        int* token_ids) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_tokens) {
        return;
    }
    
    int offset = token_offsets[tid];
    int length = token_lengths[tid];
    
    const char* token = text + offset;
    
    token_ids[tid] = binary_search_vocab(token, length, vocab, vocab_size);
}

int cpu_vocab_lookup(const char* token, int token_length, const Vocabulary* vocab) {
    int left = 0;
    int right = vocab->size - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        int cmp = strncmp(token, vocab->entries[mid].token, token_length);
        
        if (cmp == 0 && vocab->entries[mid].token_length == token_length) {
            return vocab->entries[mid].token_id;
        } else if (cmp < 0 || (cmp == 0 && token_length < vocab->entries[mid].token_length)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return UNK_TOKEN_ID;
}

void cpu_batch_vocab_lookup(const char* text, const int* token_offsets, 
                            const int* token_lengths, int num_tokens,
                            const Vocabulary* vocab, int* token_ids) {
    for (int i = 0; i < num_tokens; i++) {
        const char* token = text + token_offsets[i];
        int length = token_lengths[i];
        token_ids[i] = cpu_vocab_lookup(token, length, vocab);
    }
}

void run_benchmark(const char* text, const int* token_offsets, const int* token_lengths,
                  int num_tokens, size_t text_bytes, const Vocabulary* vocab, 
                  BenchmarkResult* result, int num_iterations) {
    result->num_tokens = num_tokens;
    result->text_bytes = text_bytes;
    
    int* cpu_token_ids = (int*)malloc(num_tokens * sizeof(int));
    int* gpu_token_ids = (int*)malloc(num_tokens * sizeof(int));
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; iter++) {
        cpu_batch_vocab_lookup(text, token_offsets, token_lengths, num_tokens, vocab, cpu_token_ids);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    result->cpu_time_ms = cpu_duration.count() / num_iterations;
    
    float cpu_time_sec = result->cpu_time_ms / 1000.0f;
    result->cpu_throughput_tokens_per_sec = num_tokens / cpu_time_sec;
    result->cpu_throughput_mbps = (text_bytes / (1024.0f * 1024.0f)) / cpu_time_sec;
    
    size_t text_size = 0;
    for (int i = 0; i < num_tokens; i++) {
        int end = token_offsets[i] + token_lengths[i];
        if (end > text_size) {
            text_size = end;
        }
    }
    
    char* d_text;
    int* d_token_offsets;
    int* d_token_lengths;
    VocabEntry* d_vocab;
    int* d_token_ids;
    
    CHECK_CUDA(cudaMalloc((void**)&d_text, text_size));
    CHECK_CUDA(cudaMalloc((void**)&d_token_offsets, num_tokens * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_token_lengths, num_tokens * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vocab, vocab->size * sizeof(VocabEntry)));
    CHECK_CUDA(cudaMalloc((void**)&d_token_ids, num_tokens * sizeof(int)));
    
    CHECK_CUDA(cudaMemcpy(d_text, text, text_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_offsets, token_offsets, num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_lengths, token_lengths, num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vocab, vocab->entries, vocab->size * sizeof(VocabEntry), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < num_iterations; iter++) {
        gpu_vocab_lookup_kernel<<<grid_size, block_size>>>(d_text, d_token_offsets, d_token_lengths,
                                                            num_tokens, d_vocab, vocab->size, d_token_ids);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float total_gpu_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_time_ms, start, stop));
    result->gpu_time_ms = total_gpu_time_ms / num_iterations;
    
    float gpu_time_sec = result->gpu_time_ms / 1000.0f;
    result->gpu_throughput_tokens_per_sec = num_tokens / gpu_time_sec;
    result->gpu_throughput_mbps = (text_bytes / (1024.0f * 1024.0f)) / gpu_time_sec;
    result->speedup = result->cpu_time_ms / result->gpu_time_ms;
    
    CHECK_CUDA(cudaMemcpy(gpu_token_ids, d_token_ids, num_tokens * sizeof(int), cudaMemcpyDeviceToHost));
    
    result->correct = true;
    for (int i = 0; i < num_tokens; i++) {
        if (cpu_token_ids[i] != gpu_token_ids[i]) {
            result->correct = false;
            break;
        }
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_text));
    CHECK_CUDA(cudaFree(d_token_offsets));
    CHECK_CUDA(cudaFree(d_token_lengths));
    CHECK_CUDA(cudaFree(d_vocab));
    CHECK_CUDA(cudaFree(d_token_ids));
    
    free(cpu_token_ids);
    free(gpu_token_ids);
}

int tokenize_text(const char* text, size_t text_size, int** token_offsets, int** token_lengths) {
    int capacity = 10000;
    *token_offsets = (int*)malloc(capacity * sizeof(int));
    *token_lengths = (int*)malloc(capacity * sizeof(int));
    
    int num_tokens = 0;
    size_t i = 0;
    
    while (i < text_size) {
        while (i < text_size && (text[i] == ' ' || text[i] == '\n' || text[i] == '\t' || 
                                  text[i] == ',' || text[i] == '.' || text[i] == '!' || 
                                  text[i] == '?' || text[i] == ';' || text[i] == ':')) {
            i++;
        }
        
        if (i >= text_size) break;
        
        size_t start = i;
        while (i < text_size && text[i] != ' ' && text[i] != '\n' && text[i] != '\t' && 
               text[i] != ',' && text[i] != '.' && text[i] != '!' && 
               text[i] != '?' && text[i] != ';' && text[i] != ':') {
            i++;
        }
        
        if (num_tokens >= capacity) {
            capacity *= 2;
            *token_offsets = (int*)realloc(*token_offsets, capacity * sizeof(int));
            *token_lengths = (int*)realloc(*token_lengths, capacity * sizeof(int));
        }
        
        (*token_offsets)[num_tokens] = start;
        (*token_lengths)[num_tokens] = i - start;
        num_tokens++;
    }
    
    return num_tokens;
}

void generate_synthetic_vocab(const char* output_path, int vocab_size) {
    FILE* fp = fopen(output_path, "w");
    if (!fp) {
        fprintf(stderr, "Failed to create vocabulary file: %s\n", output_path);
        exit(EXIT_FAILURE);
    }
    
    const char* common_words[] = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
    };
    
    int num_common = sizeof(common_words) / sizeof(common_words[0]);
    
    for (int i = 0; i < vocab_size && i < num_common; i++) {
        fprintf(fp, "%d,%s\n", i, common_words[i]);
    }
    
    for (int i = num_common; i < vocab_size; i++) {
        char token[32];
        snprintf(token, sizeof(token), "token_%d", i);
        fprintf(fp, "%d,%s\n", i, token);
    }
    
    fclose(fp);
    printf("Generated vocabulary with %d tokens at %s\n", vocab_size, output_path);
}

void generate_synthetic_text(const char* output_path, int num_words) {
    FILE* fp = fopen(output_path, "w");
    if (!fp) {
        fprintf(stderr, "Failed to create text file: %s\n", output_path);
        exit(EXIT_FAILURE);
    }
    
    const char* common_words[] = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
    };
    
    int num_common = sizeof(common_words) / sizeof(common_words[0]);
    
    for (int i = 0; i < num_words; i++) {
        fprintf(fp, "%s ", common_words[i % num_common]);
        if ((i + 1) % 15 == 0) {
            fprintf(fp, "\n");
        }
    }
    
    fclose(fp);
    printf("Generated text file with %d words at %s\n", num_words, output_path);
}

void write_csv_results(const char* filepath, BenchmarkResult* results, int num_results) {
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "Warning: Could not write CSV to %s\n", filepath);
        return;
    }
    
    fprintf(fp, "text_size_kb,num_tokens,cpu_time_ms,cpu_tokens_per_sec,cpu_mbps,gpu_time_ms,gpu_tokens_per_sec,gpu_mbps,speedup,correct\n");
    for (int i = 0; i < num_results; i++) {
        float text_kb = results[i].text_bytes / 1024.0f;
        fprintf(fp, "%.1f,%d,%.2f,%.0f,%.2f,%.2f,%.0f,%.2f,%.2f,%s\n",
                text_kb,
                results[i].num_tokens,
                results[i].cpu_time_ms, results[i].cpu_throughput_tokens_per_sec, results[i].cpu_throughput_mbps,
                results[i].gpu_time_ms, results[i].gpu_throughput_tokens_per_sec, results[i].gpu_throughput_mbps,
                results[i].speedup,
                results[i].correct ? "true" : "false");
    }
    
    fclose(fp);
    printf("\nBenchmark results written to %s\n", filepath);
}

void run_tiktoken_style_benchmark(const char* text_path, const Vocabulary* vocab) {
    printf("\n=== Tiktoken-Style Benchmark ===\n");
    printf("Methodology: Average over 100 iterations per text size\n");
    printf("Comparison baseline: tiktoken single-threaded ~3-5 MB/s, ~50-200k tokens/sec\n\n");
    
    const char* test_labels[] = {"Short (1KB)", "Medium (10KB)", "Long (100KB)", "XLong (1MB)"};
    size_t test_sizes[] = {1024, 10*1024, 100*1024, 1024*1024};
    int num_tests = 4;
    int num_iterations = 100;
    
    BenchmarkResult* results = (BenchmarkResult*)malloc(num_tests * sizeof(BenchmarkResult));
    
    FILE* fp = fopen(text_path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open text file for tiktoken benchmark\n");
        free(results);
        return;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t full_text_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* full_text = (char*)malloc(full_text_size + 1);
    fread(full_text, 1, full_text_size, fp);
    full_text[full_text_size] = '\0';
    fclose(fp);
    
    printf("%-15s | %-12s | %-18s | %-18s | %-10s\n", 
           "Text Size", "Tokens", "CPU (tokens/s)", "GPU (tokens/s)", "Speedup");
    printf("%-15s | %-12s | %-18s | %-18s | %-10s\n",
           "", "", "MB/s", "MB/s", "");
    printf("----------------|-------------|-------------------|-------------------|------------\n");
    
    for (int i = 0; i < num_tests; i++) {
        size_t test_size = (test_sizes[i] < full_text_size) ? test_sizes[i] : full_text_size;
        
        char* test_text = (char*)malloc(test_size + 1);
        memcpy(test_text, full_text, test_size);
        test_text[test_size] = '\0';
        
        int* token_offsets;
        int* token_lengths;
        int num_tokens = tokenize_text(test_text, test_size, &token_offsets, &token_lengths);
        
        run_benchmark(test_text, token_offsets, token_lengths, num_tokens, test_size,
                     vocab, &results[i], num_iterations);
        
        printf("%-15s | %7d     | %8.0f (%5.2f) | %8.0f (%5.2f) | %.2fx\n",
               test_labels[i],
               results[i].num_tokens,
               results[i].cpu_throughput_tokens_per_sec, results[i].cpu_throughput_mbps,
               results[i].gpu_throughput_tokens_per_sec, results[i].gpu_throughput_mbps,
               results[i].speedup);
        
        free(test_text);
        free(token_offsets);
        free(token_lengths);
    }
    
    free(full_text);
    
    write_csv_results("tiktoken_benchmark_results.csv", results, num_tests);
    
    printf("\nComparison to tiktoken baseline:\n");
    printf("  tiktoken single-threaded: ~3-5 MB/s, ~50-200k tokens/sec\n");
    printf("  Our CPU implementation: %.2f MB/s, %.0f tokens/sec (avg)\n", 
           results[num_tests-1].cpu_throughput_mbps, 
           results[num_tests-1].cpu_throughput_tokens_per_sec);
    printf("  Our GPU implementation: %.2f MB/s, %.0f tokens/sec (avg)\n",
           results[num_tests-1].gpu_throughput_mbps,
           results[num_tests-1].gpu_throughput_tokens_per_sec);
    
    free(results);
}

// ===========================================================================
// BPE pipeline
// ===========================================================================
//
// The vocabulary is loaded from a fixed-stride binary file produced by
// export_tiktoken_vocab.py. Entries are sorted lexicographically by their
// byte sequence so we can binary-search for the rank of an arbitrary byte
// span on the device. For tiktoken encodings, the rank doubles as the
// final token id.
//
// Pre-split text "pieces" (also produced by export_tiktoken_vocab.py) are
// fed independently into the BPE kernel. A piece is a short byte run
// (typically a word with optional leading whitespace) that BPE merges down
// to a sequence of token ids.
//
// Two kernels are provided so we can compare parallelism strategies:
//   - V1: one thread per piece. Each thread runs its own sequential merge
//         loop using a small per-thread workspace held in local memory.
//   - V2: one block per piece. Threads cooperate via shared memory: every
//         thread scores one adjacent pair, a parallel reduction picks the
//         lowest-rank pair, and thread 0 performs the merge.
// ===========================================================================

void load_bpe_ranks(const char* path, BPERank** ranks_out, int* num_ranks_out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open BPE ranks file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    unsigned int num_ranks = 0;
    unsigned int max_token_len = 0;
    if (fread(&num_ranks, sizeof(unsigned int), 1, fp) != 1 ||
        fread(&max_token_len, sizeof(unsigned int), 1, fp) != 1) {
        fprintf(stderr, "Failed to read BPE ranks header from %s\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if ((int)max_token_len != BPE_MAX_TOKEN_LEN) {
        fprintf(stderr,
                "BPE ranks file uses max_token_len=%u but binary was built "
                "with BPE_MAX_TOKEN_LEN=%d. Regenerate with the matching "
                "MAX_TOKEN_LEN.\n",
                max_token_len, BPE_MAX_TOKEN_LEN);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    BPERank* ranks = (BPERank*)malloc(num_ranks * sizeof(BPERank));
    if (!ranks) {
        fprintf(stderr, "Failed to allocate %u BPE ranks\n", num_ranks);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < num_ranks; i++) {
        unsigned int length = 0;
        if (fread(&length, sizeof(unsigned int), 1, fp) != 1 ||
            fread(ranks[i].bytes, 1, BPE_MAX_TOKEN_LEN, fp) != (size_t)BPE_MAX_TOKEN_LEN) {
            fprintf(stderr, "Truncated BPE ranks file at entry %u\n", i);
            free(ranks);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        unsigned int rank = 0;
        if (fread(&rank, sizeof(unsigned int), 1, fp) != 1) {
            fprintf(stderr, "Truncated BPE ranks file at entry %u (rank)\n", i);
            free(ranks);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        ranks[i].length = (int)length;
        ranks[i].rank = (int)rank;
    }

    fclose(fp);
    *ranks_out = ranks;
    *num_ranks_out = (int)num_ranks;
    printf("Loaded %d BPE ranks from %s\n", (int)num_ranks, path);
}

void free_bpe_ranks(BPERank* ranks) {
    if (ranks) free(ranks);
}

void load_pieces(const char* path, Pieces* pieces) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open pieces file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    unsigned int num_pieces = 0;
    unsigned int total_bytes = 0;
    if (fread(&num_pieces, sizeof(unsigned int), 1, fp) != 1 ||
        fread(&total_bytes, sizeof(unsigned int), 1, fp) != 1) {
        fprintf(stderr, "Failed to read pieces header from %s\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    pieces->num_pieces = (int)num_pieces;
    pieces->total_bytes = (int)total_bytes;
    pieces->offsets = (int*)malloc(num_pieces * sizeof(int));
    pieces->lengths = (int*)malloc(num_pieces * sizeof(int));
    pieces->blob = (unsigned char*)malloc(total_bytes);
    if (!pieces->offsets || !pieces->lengths || !pieces->blob) {
        fprintf(stderr, "Failed to allocate memory for pieces\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < num_pieces; i++) {
        unsigned int off = 0;
        unsigned int len = 0;
        if (fread(&off, sizeof(unsigned int), 1, fp) != 1 ||
            fread(&len, sizeof(unsigned int), 1, fp) != 1) {
            fprintf(stderr, "Truncated pieces file at entry %u\n", i);
            free_pieces(pieces);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        pieces->offsets[i] = (int)off;
        pieces->lengths[i] = (int)len;
    }

    if (fread(pieces->blob, 1, total_bytes, fp) != (size_t)total_bytes) {
        fprintf(stderr, "Truncated pieces file in blob section\n");
        free_pieces(pieces);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);
    printf("Loaded %d pieces (%d bytes) from %s\n",
           pieces->num_pieces, pieces->total_bytes, path);
}

void free_pieces(Pieces* pieces) {
    if (pieces->blob) { free(pieces->blob); pieces->blob = NULL; }
    if (pieces->offsets) { free(pieces->offsets); pieces->offsets = NULL; }
    if (pieces->lengths) { free(pieces->lengths); pieces->lengths = NULL; }
    pieces->num_pieces = 0;
    pieces->total_bytes = 0;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int bpe_bytes_compare(const unsigned char* a, int a_len,
                                                 const unsigned char* b, int b_len) {
    int n = a_len < b_len ? a_len : b_len;
    for (int i = 0; i < n; i++) {
        int diff = (int)a[i] - (int)b[i];
        if (diff != 0) return diff;
    }
    return a_len - b_len;
}

// Binary search for the rank of a byte span. Returns the rank if found,
// otherwise INT_MAX (signalling "no merge possible for this pair"). We
// return INT_MAX rather than -1 so reductions can treat it as "infinitely
// bad" without an extra branch.
__device__ int bpe_lookup_rank(const unsigned char* query, int query_len,
                               const BPERank* ranks, int num_ranks) {
    int lo = 0;
    int hi = num_ranks - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int cmp = bpe_bytes_compare(query, query_len,
                                    ranks[mid].bytes, ranks[mid].length);
        if (cmp == 0) return ranks[mid].rank;
        if (cmp < 0) hi = mid - 1;
        else         lo = mid + 1;
    }
    return INT_MAX;
}

// ---------------------------------------------------------------------------
// V1: one thread per piece. Sequential BPE inside each thread.
// ---------------------------------------------------------------------------
//
// We exploit the key invariant that every active token's bytes are a
// contiguous span of the original piece. So we never copy bytes around
// during merging - we only track (start, length) and a linked-list "next"
// pointer to skip over merged-away tokens.

__global__ void bpe_encode_kernel_v1(const unsigned char* blob,
                                     const int* piece_offsets,
                                     const int* piece_lengths,
                                     int num_pieces,
                                     const BPERank* ranks,
                                     int num_ranks,
                                     int* out_token_ids,
                                     int* out_token_offsets,
                                     int* out_token_counts,
                                     int max_tokens_per_piece) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pieces) return;

    const int p_off = piece_offsets[tid];
    const int p_len = piece_lengths[tid];
    if (p_len <= 0) {
        out_token_counts[tid] = 0;
        return;
    }

    const unsigned char* p_bytes = blob + p_off;

    int starts[BPE_MAX_TOKENS_PER_PIECE];
    int lengths[BPE_MAX_TOKENS_PER_PIECE];
    int next[BPE_MAX_TOKENS_PER_PIECE];

    int n = p_len < BPE_MAX_TOKENS_PER_PIECE ? p_len : BPE_MAX_TOKENS_PER_PIECE;
    for (int i = 0; i < n; i++) {
        starts[i] = i;
        lengths[i] = 1;
        next[i] = (i + 1 < n) ? (i + 1) : -1;
    }

    // BPE cannot run more than n-1 iterations (each removes one token).
    for (int iter = 0; iter < n; iter++) {
        int best_rank = INT_MAX;
        int best_pos = -1;

        int i = 0;
        while (i != -1) {
            int j = next[i];
            if (j == -1) break;

            int combined = lengths[i] + lengths[j];
            if (combined <= BPE_MAX_TOKEN_LEN) {
                int r = bpe_lookup_rank(p_bytes + starts[i], combined,
                                        ranks, num_ranks);
                if (r < best_rank) {
                    best_rank = r;
                    best_pos = i;
                }
            }
            i = j;
        }

        if (best_pos == -1) break;

        int j = next[best_pos];
        lengths[best_pos] += lengths[j];
        next[best_pos] = next[j];
    }

    int out_off = out_token_offsets[tid];
    int count = 0;
    int i = 0;
    while (i != -1 && count < max_tokens_per_piece) {
        int r = bpe_lookup_rank(p_bytes + starts[i], lengths[i],
                                ranks, num_ranks);
        out_token_ids[out_off + count] = (r == INT_MAX) ? -1 : r;
        count++;
        i = next[i];
    }
    out_token_counts[tid] = count;
}

// ---------------------------------------------------------------------------
// V2: one block per piece. Parallel pair scoring + reduction.
// ---------------------------------------------------------------------------
//
// Each thread is responsible for one source position in the piece. Per
// round it computes the rank of the pair starting at its position (or
// INT_MAX if its position is no longer active), then a tree-reduction
// across the block selects the leftmost lowest-rank pair. Thread 0 applies
// that single merge and the loop repeats.
//
// We pack (rank, position) into one 64-bit value during the reduction so
// the standard min() does both "lowest rank wins" and "leftmost ties win"
// for free.

__global__ void bpe_encode_kernel_v2(const unsigned char* blob,
                                     const int* piece_offsets,
                                     const int* piece_lengths,
                                     int num_pieces,
                                     const BPERank* ranks,
                                     int num_ranks,
                                     int* out_token_ids,
                                     int* out_token_offsets,
                                     int* out_token_counts) {
    const int piece_id = blockIdx.x;
    if (piece_id >= num_pieces) return;

    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    const int p_off = piece_offsets[piece_id];
    const int p_len = piece_lengths[piece_id];

    __shared__ int s_starts[BPE_MAX_TOKENS_PER_PIECE];
    __shared__ int s_lengths[BPE_MAX_TOKENS_PER_PIECE];
    __shared__ int s_next[BPE_MAX_TOKENS_PER_PIECE];
    __shared__ int s_active[BPE_MAX_TOKENS_PER_PIECE];
    __shared__ unsigned long long s_pair[BPE_MAX_TOKENS_PER_PIECE];
    __shared__ int s_done;

    const unsigned char* p_bytes = blob + p_off;
    const int n = p_len < BPE_MAX_TOKENS_PER_PIECE ? p_len : BPE_MAX_TOKENS_PER_PIECE;

    if (tid == 0) {
        s_done = (n <= 1) ? 1 : 0;
        if (n == 0) out_token_counts[piece_id] = 0;
    }

    for (int i = tid; i < BPE_MAX_TOKENS_PER_PIECE; i += bsz) {
        if (i < n) {
            s_starts[i] = i;
            s_lengths[i] = 1;
            s_next[i] = (i + 1 < n) ? (i + 1) : -1;
            s_active[i] = 1;
        } else {
            s_starts[i] = 0;
            s_lengths[i] = 0;
            s_next[i] = -1;
            s_active[i] = 0;
        }
    }
    __syncthreads();

    for (int iter = 0; iter < n; iter++) {
        if (s_done) break;

        for (int i = tid; i < BPE_MAX_TOKENS_PER_PIECE; i += bsz) {
            unsigned long long packed;
            if (i >= n || !s_active[i] || s_next[i] == -1) {
                packed = ((unsigned long long)INT_MAX << 32) | (unsigned int)i;
            } else {
                int j = s_next[i];
                int combined = s_lengths[i] + s_lengths[j];
                int r = INT_MAX;
                if (combined <= BPE_MAX_TOKEN_LEN) {
                    r = bpe_lookup_rank(p_bytes + s_starts[i], combined,
                                        ranks, num_ranks);
                }
                packed = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)i;
            }
            s_pair[i] = packed;
        }
        __syncthreads();

        for (int stride = BPE_MAX_TOKENS_PER_PIECE >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < stride; i += bsz) {
                unsigned long long a = s_pair[i];
                unsigned long long b = s_pair[i + stride];
                if (b < a) s_pair[i] = b;
            }
            __syncthreads();
        }

        if (tid == 0) {
            unsigned long long best = s_pair[0];
            int best_rank = (int)(best >> 32);
            int best_pos = (int)(best & 0xFFFFFFFFu);

            if (best_rank == INT_MAX) {
                s_done = 1;
            } else {
                int j = s_next[best_pos];
                s_lengths[best_pos] += s_lengths[j];
                s_next[best_pos] = s_next[j];
                s_active[j] = 0;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_off = out_token_offsets[piece_id];
        int count = 0;
        int i = 0;
        while (i != -1 && count < BPE_MAX_TOKENS_PER_PIECE) {
            int r = bpe_lookup_rank(p_bytes + s_starts[i], s_lengths[i],
                                    ranks, num_ranks);
            out_token_ids[out_off + count] = (r == INT_MAX) ? -1 : r;
            count++;
            i = s_next[i];
        }
        out_token_counts[piece_id] = count;
    }
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------

static void compute_token_offsets(const Pieces* pieces, int* offsets) {
    int running = 0;
    for (int i = 0; i < pieces->num_pieces; i++) {
        offsets[i] = running;
        running += pieces->lengths[i];
    }
}

static void append_bpe_csv_row(const char* path,
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
                               float e2e_tokens_per_sec) {
    FILE* check = fopen(path, "rb");
    int needs_header = 1;
    if (check) {
        fseek(check, 0, SEEK_END);
        if (ftell(check) > 0) needs_header = 0;
        fclose(check);
    }

    FILE* fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "Warning: could not open %s for appending\n", path);
        return;
    }
    if (needs_header) {
        // kernel_*: kernel-only time (warm buffers already on device)
        // e2e_*:    H2D pieces + kernel + D2H tokens per iteration; the one-
        //           time ranks-table copy is still amortized, matching how a
        //           real long-running tokenizer service would be deployed.
        fprintf(fp,
                "tag,kernel,threads_per_block,blocks,iterations,"
                "num_pieces,input_bytes,num_tokens,"
                "kernel_time_ms,kernel_mbps,kernel_tokens_per_sec,"
                "e2e_time_ms,e2e_mbps,e2e_tokens_per_sec\n");
    }
    fprintf(fp,
            "%s,v%d,%d,%d,%d,%d,%d,%d,%.6f,%.4f,%.2f,%.6f,%.4f,%.2f\n",
            tag ? tag : "",
            kernel_version,
            threads_per_block,
            blocks,
            iterations,
            num_pieces,
            input_bytes,
            num_tokens,
            kernel_time_ms,
            kernel_mbps,
            kernel_tokens_per_sec,
            e2e_time_ms,
            e2e_mbps,
            e2e_tokens_per_sec);
    fclose(fp);
    printf("Appended CSV row to %s\n", path);
}

static void write_token_output(const char* path,
                               const int* token_ids,
                               const int* token_offsets,
                               const int* token_counts,
                               int num_pieces) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Warning: could not open %s for writing tokens\n", path);
        return;
    }
    unsigned int n = (unsigned int)num_pieces;
    fwrite(&n, sizeof(unsigned int), 1, fp);
    for (int i = 0; i < num_pieces; i++) {
        unsigned int count = (unsigned int)token_counts[i];
        fwrite(&count, sizeof(unsigned int), 1, fp);
        if (count > 0) {
            fwrite(token_ids + token_offsets[i], sizeof(int), count, fp);
        }
    }
    fclose(fp);
    printf("Wrote token output to %s\n", path);
}

void run_bpe_benchmark(const Pieces* pieces,
                       const BPERank* h_ranks,
                       int num_ranks,
                       const BPERunConfig* config,
                       BenchmarkResult* result) {
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
        // V2: one block per piece, threads cooperate inside.
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

    // Pass 1: kernel-only timing. Inputs already on device, outputs stay on
    // device. This is what the simple-vocab benchmark also measures.
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

    // Pass 2: end-to-end per-call timing. Each iteration re-uploads the
    // pieces, runs the kernel, and copies the resulting token IDs and
    // counts back to the host. The 13.6 MB ranks table is intentionally
    // NOT re-uploaded here: in any realistic deployment that table is
    // loaded once and reused across many requests.
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

    result->num_tokens = total_tokens;
    result->text_bytes = (size_t)pieces->total_bytes;
    result->gpu_time_ms = kernel_ms;
    result->cpu_time_ms = 0.0f;
    result->gpu_throughput_tokens_per_sec = kernel_tps;
    result->gpu_throughput_mbps = kernel_mbps;
    result->cpu_throughput_tokens_per_sec = 0.0f;
    result->cpu_throughput_mbps = 0.0f;
    result->speedup = 0.0f;
    result->correct = false; // verified externally via verify_bpe.py

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

// ===========================================================================
// CLI helpers
// ===========================================================================

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --generate                 Generate synthetic vocab+text\n", prog);
    fprintf(stderr, "  %s --tiktoken                 Run legacy simple-vocab benchmark\n", prog);
    fprintf(stderr, "  %s --bpe [options]            Run BPE encode benchmark\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "BPE options:\n");
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

static int run_bpe_cli(int argc, char** argv) {
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

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--ranks") == 0 && i + 1 < argc) {
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
            fprintf(stderr, "Unknown BPE option: %s\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    BPERank* ranks = NULL;
    int num_ranks = 0;
    load_bpe_ranks(ranks_path, &ranks, &num_ranks);

    Pieces pieces = {0};
    load_pieces(pieces_path, &pieces);

    if (sweep) {
        // Single CSV file accumulates one row per (kernel, threads) config.
        // Token output is written only on the very first run so verify_bpe.py
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
            BenchmarkResult result;
            memset(&result, 0, sizeof(result));
            run_bpe_benchmark(&pieces, ranks, num_ranks, &cfg, &result);
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
            BenchmarkResult result;
            memset(&result, 0, sizeof(result));
            run_bpe_benchmark(&pieces, ranks, num_ranks, &cfg, &result);
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

        BenchmarkResult result;
        memset(&result, 0, sizeof(result));
        run_bpe_benchmark(&pieces, ranks, num_ranks, &cfg, &result);
    }

    free_pieces(&pieces);
    free_bpe_ranks(ranks);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bpe") == 0) {
        printf("GPU BPE Tokenization Benchmark\n");
        printf("================================\n");
        return run_bpe_cli(argc, argv);
    }

    printf("GPU Vocabulary Lookup Benchmark\n");
    printf("================================\n");
    printf("Comparison target: OpenAI tiktoken (single-threaded baseline)\n\n");
    
    const char* vocab_path = "data/sample_vocab.csv";
    const char* text_path = "data/sample_text.txt";
    
    if (argc > 1 && strcmp(argv[1], "--generate") == 0) {
        printf("Generating synthetic data...\n");
        generate_synthetic_vocab(vocab_path, 1000);
        generate_synthetic_text(text_path, 100000);
        printf("\nSynthetic data generated. Run again without --generate to benchmark.\n");
        return 0;
    }
    
    if (argc > 1 && strcmp(argv[1], "--tiktoken") == 0) {
        printf("Running tiktoken-style benchmark...\n\n");
        Vocabulary vocab;
        load_vocabulary(vocab_path, &vocab);
        sort_vocabulary(&vocab);
        populate_ascii_map();
        
        run_tiktoken_style_benchmark(text_path, &vocab);
        
        free_vocabulary(&vocab);
        return 0;
    }
    
    Vocabulary vocab;
    load_vocabulary(vocab_path, &vocab);
    sort_vocabulary(&vocab);
    populate_ascii_map();
    
    FILE* fp = fopen(text_path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open text file: %s\n", text_path);
        fprintf(stderr, "Run with --generate to create sample data first.\n");
        return EXIT_FAILURE;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t text_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* text = (char*)malloc(text_size + 1);
    fread(text, 1, text_size, fp);
    text[text_size] = '\0';
    fclose(fp);
    
    int* token_offsets;
    int* token_lengths;
    int num_tokens = tokenize_text(text, text_size, &token_offsets, &token_lengths);
    
    printf("Tokenized %d tokens from text (%.1f KB)\n", num_tokens, text_size / 1024.0f);
    printf("Running quick benchmark (single iteration)...\n\n");
    
    BenchmarkResult result;
    run_benchmark(text, token_offsets, token_lengths, num_tokens, text_size, &vocab, &result, 1);
    
    printf("Results:\n");
    printf("  Tokens processed: %d\n", result.num_tokens);
    printf("  Text size: %.1f KB\n", result.text_bytes / 1024.0f);
    printf("  CPU: %.2f ms (%.0f tokens/sec, %.2f MB/s)\n", 
           result.cpu_time_ms, result.cpu_throughput_tokens_per_sec, result.cpu_throughput_mbps);
    printf("  GPU: %.2f ms (%.0f tokens/sec, %.2f MB/s)\n", 
           result.gpu_time_ms, result.gpu_throughput_tokens_per_sec, result.gpu_throughput_mbps);
    printf("  Speedup: %.2fx\n", result.speedup);
    printf("  Correctness: %s\n", result.correct ? "PASS" : "FAIL");
    
    printf("\nTiktoken baseline comparison:\n");
    printf("  tiktoken (single-threaded): ~3-5 MB/s, ~50-200k tokens/sec\n");
    printf("  Our GPU implementation: %.2f MB/s, %.0f tokens/sec\n",
           result.gpu_throughput_mbps, result.gpu_throughput_tokens_per_sec);
    
    printf("\nRun with --tiktoken for comprehensive multi-size benchmark\n");
    printf("Run with --generate to regenerate test data\n");
    
    BenchmarkResult results[1] = {result};
    write_csv_results("benchmark_results.csv", results, 1);
    
    free(text);
    free(token_offsets);
    free(token_lengths);
    free_vocabulary(&vocab);
    
    return 0;
}
