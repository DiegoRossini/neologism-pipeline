#!/usr/bin/env python3

import argparse
import bz2
import re
import subprocess
import sys
from pathlib import Path

DUMP_URL = "https://archive.org/download/enwiktionary-20141101/enwiktionary-20141101-pages-articles.xml.bz2"


def download_dump(dump_file):
    if dump_file.exists():
        print(f"Dump already exists: {dump_file}")
        print(f"Size: {dump_file.stat().st_size / (1024**2):.1f} MB")
        return True

    print(f"Downloading Wiktionary November 2014 dump...")
    print(f"URL: {DUMP_URL}")

    cmd = ["wget", "-c", "-O", str(dump_file), DUMP_URL]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Download failed!")
        return False

    print(f"Download complete: {dump_file}")
    return True


def extract_titles(dump_file):
    print(f"Extracting ENGLISH titles from {dump_file}...")

    titles = set()
    title_pattern = re.compile(r'<title>([^<]+)</title>')
    text_start_pattern = re.compile(r'<text[^>]*>')
    text_end_pattern = re.compile(r'</text>')

    bytes_read = 0
    file_size = dump_file.stat().st_size

    current_title = None
    in_text = False
    text_buffer = []

    with bz2.open(dump_file, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            bytes_read += len(line.encode('utf-8'))

            title_match = title_pattern.search(line)
            if title_match:
                current_title = title_match.group(1).strip()
                if ':' in current_title:
                    current_title = None
                continue

            if text_start_pattern.search(line):
                in_text = True
                text_buffer = [line]
                continue

            if in_text:
                text_buffer.append(line)

                if text_end_pattern.search(line):
                    in_text = False
                    full_text = ''.join(text_buffer)

                    if current_title and '==English==' in full_text:
                        titles.add(current_title.lower())

                    current_title = None
                    text_buffer = []

            if bytes_read % (50 * 1024 * 1024) < 1000:
                pct = (bytes_read / file_size) * 100
                print(f"  Progress: {pct:.1f}%, English titles: {len(titles):,}")

    print(f"\nTotal unique English titles: {len(titles):,}")
    return titles


def save_vocab(titles, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for title in sorted(titles):
            f.write(f"{title}\n")

    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Wiktionary Pre-2015 Vocabulary Extraction")
    parser.add_argument("--output", type=str, required=True, help="Output vocabulary file path")
    parser.add_argument("--dump-dir", type=str, required=True, help="Directory to store/find the Wiktionary dump")
    args = parser.parse_args()

    dump_file = Path(args.dump_dir) / "enwiktionary-20141101-pages-articles.xml.bz2"
    output_file = Path(args.output)

    print("=" * 60)
    print("Wiktionary Pre-2015 Vocabulary Extraction")
    print("=" * 60)

    print("\n[Step 1/2] Downloading dump...")
    if not download_dump(dump_file):
        sys.exit(1)

    print("\n[Step 2/2] Extracting titles...")
    titles = extract_titles(dump_file)

    save_vocab(titles, output_file)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Vocabulary file: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
