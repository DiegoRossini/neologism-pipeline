#!/usr/bin/env python3

import argparse
import bz2
import re
import subprocess
import sys
from pathlib import Path

DUMP_URL = "https://archive.org/download/enwiki-20141106/enwiki-20141106-pages-articles.xml.bz2"


def download_dump(dump_file):
    if dump_file.exists():
        print(f"Dump already exists: {dump_file}")
        print(f"Size: {dump_file.stat().st_size / (1024**3):.2f} GB")
        return True

    print(f"Downloading Wikipedia November 2014 dump...")
    print(f"URL: {DUMP_URL}")
    print("This is ~12GB, may take a while...")

    cmd = ["wget", "-c", "-O", str(dump_file), DUMP_URL]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Download failed!")
        return False

    print(f"Download complete: {dump_file}")
    return True


def is_disambiguation_page(page_content):
    disambig_templates = [
        r'\{\{disambig',
        r'\{\{disambiguation',
        r'\{\{dab\}\}',
        r'\{\{hndis',
        r'\{\{geodis',
        r'\{\{surname',
        r'\{\{given name',
        r'\{\{ship index',
        r'\{\{set index',
        r'\{\{letter-number',
        r'\{\{mathdab',
        r'\{\{numberdis',
        r'\{\{letter disambig',
        r'\{\{species latin name disambiguation',
        r'\{\{taxonomic authorities disambiguation',
        r'\{\{callsign disambig',
        r'\{\{airport disambig',
        r'\{\{school disambig',
        r'\{\{hospital disambig',
        r'\{\{mil-unit-dis',
        r'\{\{SIA\}\}',
    ]

    content_lower = page_content.lower()

    for template in disambig_templates:
        if re.search(template, content_lower):
            return True

    if 'category:disambiguation' in content_lower:
        return True
    if 'category:all disambiguation pages' in content_lower:
        return True
    if 'category:all set index articles' in content_lower:
        return True

    return False


def extract_titles(dump_file):
    print(f"Extracting titles from {dump_file}...", flush=True)
    print("Excluding redirect pages and disambiguation pages...", flush=True)

    titles = set()
    redirects_skipped = 0
    disambig_skipped = 0

    title_pattern = re.compile(r'<title>([^<]+)</title>')
    redirect_pattern = re.compile(r'<redirect\s')
    page_start_pattern = re.compile(r'<page>')
    page_end_pattern = re.compile(r'</page>')

    bytes_read = 0
    file_size = dump_file.stat().st_size
    last_pct = -1

    in_page = False
    page_lines = []

    with bz2.open(dump_file, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            bytes_read += len(line.encode('utf-8'))

            if page_start_pattern.search(line):
                in_page = True
                page_lines = [line]
                continue

            if in_page:
                page_lines.append(line)

                if page_end_pattern.search(line):
                    page_content = ''.join(page_lines)

                    if redirect_pattern.search(page_content):
                        redirects_skipped += 1
                        in_page = False
                        page_lines = []
                        continue

                    if is_disambiguation_page(page_content):
                        disambig_skipped += 1
                        in_page = False
                        page_lines = []
                        continue

                    match = title_pattern.search(page_content)
                    if match:
                        title = match.group(1).strip()

                        if ':' not in title:
                            title_lower = title.lower()
                            titles.add(title_lower)

                    in_page = False
                    page_lines = []

            pct = int((bytes_read / file_size) * 100)
            if pct > last_pct:
                last_pct = pct
                print(f"  Progress: {pct}%, titles: {len(titles):,}, redirects: {redirects_skipped:,}, disambig: {disambig_skipped:,}", flush=True)

    print(f"\nTotal unique titles: {len(titles):,}", flush=True)
    print(f"Redirects skipped: {redirects_skipped:,}", flush=True)
    print(f"Disambiguation pages skipped: {disambig_skipped:,}", flush=True)
    return titles


def save_vocab(titles, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for title in sorted(titles):
            f.write(f"{title}\n")

    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Wikipedia Pre-2015 Titles Extraction")
    parser.add_argument("--output", type=str, required=True, help="Output vocabulary file path")
    parser.add_argument("--dump-dir", type=str, required=True, help="Directory to store/find the Wikipedia dump")
    args = parser.parse_args()

    dump_file = Path(args.dump_dir) / "enwiki-20141106-pages-articles.xml.bz2"
    output_file = Path(args.output)

    print("=" * 60)
    print("Wikipedia Pre-2015 Titles Extraction")
    print("(No Redirects, No Disambiguation Pages)")
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
