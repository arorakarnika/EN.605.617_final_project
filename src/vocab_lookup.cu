#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <algorithm>
#include "vocab_lookup.h"

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
                  int num_tokens, const Vocabulary* vocab, BenchmarkResult* result) {
    result->num_tokens = num_tokens;
    
    int* cpu_token_ids = (int*)malloc(num_tokens * sizeof(int));
    int* gpu_token_ids = (int*)malloc(num_tokens * sizeof(int));
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_batch_vocab_lookup(text, token_offsets, token_lengths, num_tokens, vocab, cpu_token_ids);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    result->cpu_time_ms = cpu_duration.count();
    result->cpu_throughput = (num_tokens / result->cpu_time_ms) * 1000.0f;
    
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
    gpu_vocab_lookup_kernel<<<grid_size, block_size>>>(d_text, d_token_offsets, d_token_lengths,
                                                        num_tokens, d_vocab, vocab->size, d_token_ids);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    CHECK_CUDA(cudaEventElapsedTime(&result->gpu_time_ms, start, stop));
    result->gpu_throughput = (num_tokens / result->gpu_time_ms) * 1000.0f;
    
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
    
    fprintf(fp, "num_tokens,cpu_time_ms,cpu_throughput,gpu_time_ms,gpu_throughput,speedup,correct\n");
    for (int i = 0; i < num_results; i++) {
        float speedup = results[i].cpu_time_ms / results[i].gpu_time_ms;
        fprintf(fp, "%d,%.2f,%.0f,%.2f,%.0f,%.2f,%s\n",
                results[i].num_tokens,
                results[i].cpu_time_ms, results[i].cpu_throughput,
                results[i].gpu_time_ms, results[i].gpu_throughput,
                speedup,
                results[i].correct ? "true" : "false");
    }
    
    fclose(fp);
    printf("\nBenchmark results written to %s\n", filepath);
}

int main(int argc, char** argv) {
    printf("GPU Vocabulary Lookup Benchmark\n");
    printf("================================\n\n");
    
    const char* vocab_path = "data/sample_vocab.csv";
    const char* text_path = "data/sample_text.txt";
    
    if (argc > 1 && strcmp(argv[1], "--generate") == 0) {
        printf("Generating synthetic data...\n");
        generate_synthetic_vocab(vocab_path, 1000);
        generate_synthetic_text(text_path, 10000);
        printf("\nSynthetic data generated. Run again without --generate to benchmark.\n");
        return 0;
    }
    
    Vocabulary vocab;
    load_vocabulary(vocab_path, &vocab);
    sort_vocabulary(&vocab);
    
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
    
    printf("Tokenized %d tokens from text\n", num_tokens);
    printf("Running benchmark...\n\n");
    
    BenchmarkResult result;
    run_benchmark(text, token_offsets, token_lengths, num_tokens, &vocab, &result);
    
    printf("Results:\n");
    printf("  Tokens processed: %d\n", result.num_tokens);
    printf("  CPU time: %.2f ms (%.0f tokens/sec)\n", result.cpu_time_ms, result.cpu_throughput);
    printf("  GPU time: %.2f ms (%.0f tokens/sec)\n", result.gpu_time_ms, result.gpu_throughput);
    printf("  Speedup: %.2fx\n", result.cpu_time_ms / result.gpu_time_ms);
    printf("  Correctness: %s\n", result.correct ? "PASS" : "FAIL");
    
    BenchmarkResult results[1] = {result};
    write_csv_results("benchmark_results.csv", results, 1);
    
    free(text);
    free(token_offsets);
    free(token_lengths);
    free_vocabulary(&vocab);
    
    return 0;
}
