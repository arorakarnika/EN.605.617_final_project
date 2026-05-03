"""
Download a public-domain text from Project Gutenberg and emit truncated
versions at several sizes for scaling experiments.

Defaults to Pride and Prejudice (about 700 KB after header/footer stripping),
which is more than enough to saturate the GPU BPE pipeline with all four
standard size buckets (1 KB, 10 KB, 100 KB, 1 MB). The Gutenberg cache URLs
are stable and require no authentication.

Outputs (with default flags):
    data/corpus.txt           full cleaned book
    data/corpus_1KB.txt       first 1024 bytes
    data/corpus_10KB.txt      first 10240 bytes
    data/corpus_100KB.txt     first 102400 bytes
    data/corpus_1MB.txt       first 1048576 bytes (or full text if smaller)
"""

import argparse
import sys
import urllib.request
from pathlib import Path


BOOKS = {
    "pride_and_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "moby_dick":           "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "alice":               "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "frankenstein":        "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "tale_of_two_cities":  "https://www.gutenberg.org/cache/epub/98/pg98.txt",
}


def strip_gutenberg_boilerplate(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    if start_idx != -1:
        nl = text.find("\n", start_idx)
        if nl != -1:
            text = text[nl + 1:]

    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    return text.strip() + "\n"


def download(url):
    print("Downloading {}".format(url))
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def write_truncations(text, output_path, sizes):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    full_bytes = len(text.encode("utf-8"))
    print("Wrote {} ({} bytes)".format(output_path, full_bytes))

    stem = output_path.stem
    suffix = output_path.suffix or ".txt"
    parent = output_path.parent

    for size in sizes:
        if size <= 0:
            continue
        label = format_size_label(size)
        truncated_path = parent / "{}_{}{}".format(stem, label, suffix)
        truncated = truncate_to_bytes(text, size)
        truncated_path.write_text(truncated, encoding="utf-8")
        actual = len(truncated.encode("utf-8"))
        print("Wrote {} ({} bytes)".format(truncated_path, actual))


def format_size_label(num_bytes):
    if num_bytes >= 1024 * 1024 and num_bytes % (1024 * 1024) == 0:
        return "{}MB".format(num_bytes // (1024 * 1024))
    if num_bytes >= 1024 and num_bytes % 1024 == 0:
        return "{}KB".format(num_bytes // 1024)
    return "{}B".format(num_bytes)


def truncate_to_bytes(text, target_bytes):
    encoded = text.encode("utf-8")
    if len(encoded) <= target_bytes:
        return text
    truncated = encoded[:target_bytes]
    # Drop a possibly half-formed UTF-8 sequence at the cut point so the file
    # stays valid UTF-8.
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]
    if truncated and (truncated[-1] & 0xE0) == 0xC0:
        truncated = truncated[:-1]
    elif len(truncated) >= 2 and (truncated[-2] & 0xF0) == 0xE0:
        truncated = truncated[:-2]
    elif len(truncated) >= 3 and (truncated[-3] & 0xF8) == 0xF0:
        truncated = truncated[:-3]
    return truncated.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Fetch a public-domain corpus for BPE benchmarking")
    parser.add_argument("--book", default="pride_and_prejudice", choices=sorted(BOOKS.keys()),
                        help="Which Gutenberg book to download")
    parser.add_argument("--output", default="data/corpus.txt", help="Where to write the cleaned full text")
    parser.add_argument("--sizes", default="1024,10240,102400,1048576",
                        help="Comma-separated list of byte sizes for truncated copies (0 disables)")
    args = parser.parse_args()

    raw = download(BOOKS[args.book])
    cleaned = strip_gutenberg_boilerplate(raw)

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    write_truncations(cleaned, args.output, sizes)


if __name__ == "__main__":
    sys.exit(main())
