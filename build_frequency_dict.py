#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.filtering_utils import is_valid_candidate, load_stopwords
from config import BASE_DIR, VOCAB_DIR, TOKEN_COUNTS_FILE

OUTPUT_FILE = VOCAB_DIR / "reddit_pre2015_frequencies.txt"

SYMSPELL_DICT = VOCAB_DIR / "symspell_frequency_dict.txt"
SYMSPELL_BACKUP = VOCAB_DIR / "symspell_frequency_dict.txt.bak"

CUTOFF_YEAR = 2015

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def count_lines(filepath):
    logging.info(f"Counting lines in {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    logging.info(f"Total lines: {count:,}")
    return count

def extract_pre2015_frequencies():
    logging.info(f"Reading from: {TOKEN_COUNTS_FILE}")
    logging.info(f"Output to: {OUTPUT_FILE}")
    logging.info(f"Cutoff year: {CUTOFF_YEAR}")

    total_lines = count_lines(TOKEN_COUNTS_FILE)

    pre2015_tokens = {}
    tokens_with_pre2015 = 0
    tokens_without_pre2015 = 0

    logging.info("Extracting pre-2015 frequencies...")
    with open(TOKEN_COUNTS_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing tokens"):
            try:
                data = json.loads(line.strip())
                token = data.get("token", "")
                counts = data.get("counts", {})

                if not token or not counts:
                    continue

                pre2015_count = 0
                for date_str, count in counts.items():
                    try:
                        year = int(date_str[:4])
                        if year < CUTOFF_YEAR:
                            pre2015_count += count
                    except (ValueError, IndexError):
                        continue

                if pre2015_count > 0:
                    pre2015_tokens[token] = pre2015_count
                    tokens_with_pre2015 += 1
                else:
                    tokens_without_pre2015 += 1

            except json.JSONDecodeError:
                continue

    logging.info(f"Tokens with pre-2015 data: {tokens_with_pre2015:,}")
    logging.info(f"Tokens without pre-2015 data (post-2015 only): {tokens_without_pre2015:,}")

    logging.info(f"Writing frequency dictionary to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for token, freq in tqdm(sorted(pre2015_tokens.items()), desc="Writing"):
            f.write(f"{token}\t{freq}\n")

    logging.info(f"Done! Written {len(pre2015_tokens):,} tokens with frequencies.")

    frequencies = list(pre2015_tokens.values())
    frequencies.sort(reverse=True)

    logging.info("\nFrequency distribution:")
    logging.info(f"  Max frequency: {frequencies[0]:,}")
    logging.info(f"  Min frequency: {frequencies[-1]:,}")
    logging.info(f"  Median frequency: {frequencies[len(frequencies)//2]:,}")

    logging.info("\nTop 20 tokens by frequency:")
    top_tokens = sorted(pre2015_tokens.items(), key=lambda x: x[1], reverse=True)[:20]
    for token, freq in top_tokens:
        logging.info(f"  {token}: {freq:,}")

    return OUTPUT_FILE

def rebuild_symspell_dict():
    logging.info("=" * 60)
    logging.info("Rebuilding SymSpell Frequency Dictionary")
    logging.info("=" * 60)

    if not SYMSPELL_BACKUP.exists():
        logging.error(f"Backup file not found: {SYMSPELL_BACKUP}")
        return None

    logging.info("Loading stopwords...")
    stopwords = load_stopwords()
    logging.info(f"Loaded {len(stopwords)} stopwords")

    logging.info(f"\n[Step 1/4] Loading backup: {SYMSPELL_BACKUP}")
    backup_dict = {}
    with open(SYMSPELL_BACKUP, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading backup"):
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    freq = int(parts[1])
                    backup_dict[word] = freq
                except ValueError:
                    continue
    logging.info(f"Loaded {len(backup_dict):,} entries from backup")

    logging.info(f"\n[Step 2/4] Filtering with pattern rules...")
    filtered_dict = {}
    rejected = 0
    for word, freq in tqdm(backup_dict.items(), desc="Filtering"):
        if is_valid_candidate(word, stopwords):
            filtered_dict[word] = freq
        else:
            rejected += 1

    logging.info(f"After filtering: {len(filtered_dict):,} entries")
    logging.info(f"Rejected: {rejected:,} entries")

    logging.info(f"\n[Step 3/4] Loading built-in SymSpell dictionary...")
    try:
        from symspellpy import SymSpell
        sym_spell = SymSpell()
        import pkg_resources
        dict_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        builtin_dict = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    try:
                        freq = int(parts[1])
                        builtin_dict[word] = freq
                    except ValueError:
                        continue
        logging.info(f"Loaded {len(builtin_dict):,} entries from built-in dictionary")

        added_from_builtin = 0
        for word, freq in builtin_dict.items():
            if word not in filtered_dict and is_valid_candidate(word, stopwords):
                filtered_dict[word] = freq
                added_from_builtin += 1
        logging.info(f"Added {added_from_builtin:,} entries from built-in dictionary")

    except Exception as e:
        logging.warning(f"Could not load built-in SymSpell dictionary: {e}")
        logging.warning("Continuing without merging...")

    logging.info(f"\n[Step 4/4] Saving to: {SYMSPELL_DICT}")
    with open(SYMSPELL_DICT, 'w', encoding='utf-8') as f:
        for word, freq in tqdm(sorted(filtered_dict.items()), desc="Writing"):
            f.write(f"{word}\t{freq}\n")

    logging.info(f"\nDone! Wrote {len(filtered_dict):,} entries to {SYMSPELL_DICT}")

    frequencies = list(filtered_dict.values())
    frequencies.sort(reverse=True)

    logging.info("\nFrequency distribution:")
    logging.info(f"  Max frequency: {frequencies[0]:,}")
    logging.info(f"  Min frequency: {frequencies[-1]:,}")
    logging.info(f"  Median frequency: {frequencies[len(frequencies)//2]:,}")

    return SYMSPELL_DICT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or rebuild SymSpell frequency dictionary"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild symspell_frequency_dict.txt from backup with pattern filtering"
    )
    args = parser.parse_args()

    setup_logging()

    if args.rebuild:
        rebuild_symspell_dict()
    else:
        extract_pre2015_frequencies()
