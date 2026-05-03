"""
Export tiktoken artifacts for the GPU BPE pipeline.

Two outputs:
  1. data/bpe_ranks.bin
       Binary table of (token_bytes -> rank) entries, sorted lexicographically
       by token_bytes. The GPU loads this into a fixed-stride array and
       performs binary search to look up the rank of an arbitrary byte
       sequence (the rank is also the token id for tiktoken encodings).

  2. data/pieces.bin (only when --text is provided)
       Pre-split text pieces produced by tiktoken's regex pattern. Each piece
       is fed independently into the GPU BPE kernel.

Binary layouts
--------------
bpe_ranks.bin
    uint32 num_ranks
    uint32 max_token_len
    repeat num_ranks times:
        uint32 length
        bytes  token_bytes (padded to max_token_len)
        uint32 rank

pieces.bin
    uint32 num_pieces
    uint32 total_bytes
    repeat num_pieces times:
        uint32 offset_in_blob
        uint32 length
    bytes blob[total_bytes]
"""

import argparse
import struct
import sys
from pathlib import Path

import regex
import tiktoken


MAX_TOKEN_LEN = 128


def export_bpe_ranks(encoding_name, output_path):
    enc = tiktoken.get_encoding(encoding_name)
    mergeable = enc._mergeable_ranks
    print("Loaded {} mergeable tokens from {}".format(len(mergeable), encoding_name))

    too_long = [b for b in mergeable if len(b) > MAX_TOKEN_LEN]
    if too_long:
        raise RuntimeError(
            "Found {} tokens longer than MAX_TOKEN_LEN={}".format(
                len(too_long), MAX_TOKEN_LEN
            )
        )

    sorted_items = sorted(mergeable.items(), key=lambda kv: kv[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", len(sorted_items), MAX_TOKEN_LEN))
        for token_bytes, rank in sorted_items:
            length = len(token_bytes)
            padded = token_bytes + b"\x00" * (MAX_TOKEN_LEN - length)
            f.write(struct.pack("<I", length))
            f.write(padded)
            f.write(struct.pack("<I", rank))

    print("Wrote {} ranks to {}".format(len(sorted_items), output_path))
    return output_path


def export_pieces(encoding_name, text_path, output_path, max_piece_bytes):
    enc = tiktoken.get_encoding(encoding_name)
    pattern = regex.compile(enc._pat_str)

    text = Path(text_path).read_text(encoding="utf-8")

    pieces = []
    skipped_long = 0
    for match in pattern.finditer(text):
        piece = match.group(0).encode("utf-8")
        if len(piece) == 0:
            continue
        if len(piece) > max_piece_bytes:
            skipped_long += 1
            continue
        pieces.append(piece)

    blob = b"".join(pieces)
    offsets = []
    cursor = 0
    for piece in pieces:
        offsets.append(cursor)
        cursor += len(piece)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", len(pieces), len(blob)))
        for offset, piece in zip(offsets, pieces):
            f.write(struct.pack("<II", offset, len(piece)))
        f.write(blob)

    print(
        "Wrote {} pieces ({} bytes) from {} to {}".format(
            len(pieces), len(blob), text_path, output_path
        )
    )
    if skipped_long:
        print(
            "Note: skipped {} pieces longer than {} bytes (rare)".format(
                skipped_long, max_piece_bytes
            )
        )

    sample = pieces[:5]
    print("First pieces:")
    for p in sample:
        try:
            shown = p.decode("utf-8")
        except UnicodeDecodeError:
            shown = repr(p)
        print("  len={} value={!r}".format(len(p), shown))

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export tiktoken vocab and pieces for GPU BPE")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    parser.add_argument("--ranks-out", default="data/bpe_ranks.bin", help="Output binary ranks file")
    parser.add_argument("--text", default=None, help="Optional text file to pre-split into pieces")
    parser.add_argument("--pieces-out", default="data/pieces.bin", help="Output binary pieces file")
    parser.add_argument("--max-piece-bytes", type=int, default=256, help="Skip pieces longer than this")
    parser.add_argument("--skip-ranks", action="store_true", help="Do not regenerate ranks file")
    args = parser.parse_args()

    if not args.skip_ranks:
        export_bpe_ranks(args.encoding, args.ranks_out)

    if args.text:
        export_pieces(args.encoding, args.text, args.pieces_out, args.max_piece_bytes)


if __name__ == "__main__":
    sys.exit(main())
