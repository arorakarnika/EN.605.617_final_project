"""
Verify GPU BPE output against tiktoken.

Reads the binary token output produced by ./bpe_tokenizer.exe and
compares each piece's token ids to what tiktoken produces from the same
input text. Reports per-piece mismatches and an overall pass/fail summary.

Layout of the GPU token file (matches write_token_output() in src/bpe_io.cu):
    uint32 num_pieces
    repeat num_pieces times:
        uint32 count
        int32  token_ids[count]
"""

import argparse
import struct
import sys
from pathlib import Path

import regex
import tiktoken


def read_gpu_tokens(path):
    with open(path, "rb") as f:
        (num_pieces,) = struct.unpack("<I", f.read(4))
        all_tokens = []
        for _ in range(num_pieces):
            (count,) = struct.unpack("<I", f.read(4))
            tokens = list(struct.unpack("<{}i".format(count), f.read(4 * count))) if count > 0 else []
            all_tokens.append(tokens)
    return all_tokens


def split_pieces(text, encoding_name, max_piece_bytes):
    enc = tiktoken.get_encoding(encoding_name)
    pattern = regex.compile(enc._pat_str)
    pieces = []
    skipped = 0
    for match in pattern.finditer(text):
        piece_bytes = match.group(0).encode("utf-8")
        if len(piece_bytes) == 0:
            continue
        if len(piece_bytes) > max_piece_bytes:
            skipped += 1
            continue
        pieces.append(piece_bytes)
    return pieces, skipped


def main():
    parser = argparse.ArgumentParser(description="Verify GPU BPE output vs tiktoken")
    parser.add_argument("--text", required=True, help="Original text file used for pre-splitting")
    parser.add_argument("--gpu-tokens", default="data/gpu_tokens.bin")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--max-piece-bytes", type=int, default=256)
    parser.add_argument("--max-mismatches", type=int, default=10,
                        help="Stop printing after this many mismatched pieces")
    args = parser.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    gpu_pieces = read_gpu_tokens(args.gpu_tokens)
    expected_pieces, skipped = split_pieces(text, args.encoding, args.max_piece_bytes)

    if len(gpu_pieces) != len(expected_pieces):
        print("Piece count mismatch: GPU={} expected={}".format(
            len(gpu_pieces), len(expected_pieces)))
        sys.exit(2)

    enc = tiktoken.get_encoding(args.encoding)
    if not hasattr(enc, "_encode_single_piece"):
        print("This tiktoken version lacks _encode_single_piece; please upgrade tiktoken")
        sys.exit(2)

    total = len(expected_pieces)
    mismatches = 0
    shown = 0
    total_tokens_gpu = 0
    total_tokens_ref = 0

    for idx, (gpu_tokens, piece_bytes) in enumerate(zip(gpu_pieces, expected_pieces)):
        ref_tokens = enc._encode_single_piece(piece_bytes)
        total_tokens_gpu += len(gpu_tokens)
        total_tokens_ref += len(ref_tokens)
        if list(gpu_tokens) != list(ref_tokens):
            mismatches += 1
            if shown < args.max_mismatches:
                try:
                    shown_text = piece_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    shown_text = repr(piece_bytes)
                print("Mismatch at piece {} (len={} bytes={!r}):".format(
                    idx, len(piece_bytes), shown_text))
                print("  GPU: {}".format(gpu_tokens))
                print("  Ref: {}".format(ref_tokens))
                shown += 1

    print("")
    print("Pieces compared: {}".format(total))
    print("Pieces skipped (too long): {}".format(skipped))
    print("Total GPU tokens: {}".format(total_tokens_gpu))
    print("Total reference tokens: {}".format(total_tokens_ref))
    print("Mismatched pieces: {} ({:.2f}%)".format(
        mismatches, 100.0 * mismatches / max(total, 1)))

    if mismatches == 0:
        print("PASS: GPU output matches tiktoken")
        sys.exit(0)
    else:
        print("FAIL: see mismatches above")
        sys.exit(1)


if __name__ == "__main__":
    main()
