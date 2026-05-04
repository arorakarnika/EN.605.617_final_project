"""
Download public-domain texts from Project Gutenberg for BPE benchmarking.

Default: download every book in BOOKS, strip boilerplate, concatenate with
clear separators, then write the full text plus truncated copies at large
byte sizes (defaults start at 1 MiB for GPU-meaningful runs).

Single-book mode: pass --book <key> to fetch only one title.

Outputs (default --output data/corpus.txt, default --sizes):
    data/corpus.txt           full combined cleaned text
    data/corpus_1MB.txt       first 1048576 bytes (if default sizes include 1MB)
    ...                       other truncations per --sizes
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
    "complete_works_of_shakespeare": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    "war_and_peace": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "les_miserables": "https://www.gutenberg.org/cache/epub/135/pg135.txt",
    "bible": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "count_of_monte_cristo": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    "brothers_karamazov": "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
    "city_of_god": "https://www.gutenberg.org/cache/epub/45304/pg45304.txt",
    "middlemarch": "https://www.gutenberg.org/cache/epub/145/pg145.txt",
    "complete_chaucer": "https://www.gutenberg.org/cache/epub/22120/pg22120.txt",
    "life_of_charles_dickens": "https://www.gutenberg.org/cache/epub/25851/pg25851.txt",

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
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_one_book(key):
    raw = download(BOOKS[key])
    return strip_gutenberg_boilerplate(raw)


def fetch_combined_corpus():
    chunks = []
    for key in sorted(BOOKS.keys()):
        cleaned = fetch_one_book(key)
        chunks.append("\n\n=== {} ===\n\n".format(key))
        chunks.append(cleaned)
        print("Added {} ({} bytes utf-8)".format(key, len(cleaned.encode("utf-8"))))
    return "".join(chunks)


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
    parser = argparse.ArgumentParser(
        description="Fetch Gutenberg text(s) for BPE benchmarking. "
                    "Default: combine all books in BOOKS. Use --book for a single title."
    )
    parser.add_argument(
        "--book",
        default=None,
        choices=sorted(BOOKS.keys()),
        help="Fetch only this book (omit to combine every book in BOOKS)",
    )
    parser.add_argument("--output", default="data/corpus.txt", help="Where to write the cleaned full text")
    parser.add_argument(
        "--sizes",
        default="1048576,5242880,10485760,20971520",
        help="Comma-separated byte sizes for truncated copies (0 disables). Default: 1,5,10,20 MB",
    )
    args = parser.parse_args()

    if args.book:
        print("Single-book mode: {}".format(args.book))
        text = fetch_one_book(args.book)
    else:
        print("Combining all {} books from BOOKS".format(len(BOOKS)))
        text = fetch_combined_corpus()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    write_truncations(text, args.output, sizes)


if __name__ == "__main__":
    sys.exit(main())
