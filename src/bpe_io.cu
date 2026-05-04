// Host-side I/O for the BPE pipeline.
//
//   - Reads the binary ranks file produced by src/export_tiktoken_vocab.py
//     (fixed-stride layout so a sorted lex-key binary search works on GPU).
//   - Reads the binary pieces file produced by src/export_tiktoken_vocab.py.
//   - Writes the GPU's per-piece token output for src/verify_bpe.py to read.
//   - Appends one timing row per benchmark configuration to a CSV.

#include <stdio.h>
#include <stdlib.h>
#include "bpe.h"

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

void write_token_output(const char* path,
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
