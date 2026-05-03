// CUDA kernels and device helpers for the GPU BPE encoder.
//
// We exploit the key invariant that every active token's bytes are a
// contiguous span of the original piece. So we never copy bytes around
// during merging - we only track (start, length) and a linked-list "next"
// pointer to skip over merged-away tokens.
//
//   V1: one thread per piece. Sequential merge loop in per-thread storage.
//   V2: one block per piece. Threads cooperate via shared memory: every
//       thread scores one adjacent pair, a parallel reduction picks the
//       lowest-rank pair (leftmost wins ties via packed (rank, position)),
//       and thread 0 applies the single merge.

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "bpe.h"

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
