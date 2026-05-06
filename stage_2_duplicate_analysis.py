#!/usr/bin/env python3

import pandas as pd
import hashlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import sys
import logging
import subprocess

from config import (
    OUTPUT_DIR,
    LOG_DIR,
    CHECKPOINTS_DIR,
    SCRIPTS_DIR,
    DUPLICATE_IDS_FILE,
    detect_text_column,
    get_all_csv_files,
)

N_WORKERS = 8
CHECKPOINT_EVERY = 2000

STAGE2_CHECKPOINT_DIR = SCRIPTS_DIR / "stage2_checkpoints"
TEMP_FILE = STAGE2_CHECKPOINT_DIR / "all_hashes.tsv"
SORTED_FILE = STAGE2_CHECKPOINT_DIR / "all_hashes_sorted.tsv"


def setup_logging():
    log_file = LOG_DIR / "stage_2_duplicate_analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def hash_text(text):
    if pd.isna(text) or text == "":
        return None
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()


def process_file(filepath):
    lines = []
    try:
        text_col = detect_text_column(str(filepath))
        if text_col is None:
            return filepath.name, lines

        df = pd.read_csv(
            filepath,
            compression='gzip',
            on_bad_lines='skip',
            usecols=['author', 'date', text_col, 'id', 'subreddit'],
            dtype=str
        )

        for _, row in df.iterrows():
            text = row.get(text_col, '')
            if pd.isna(text) or text == '' or len(str(text).strip()) < 10:
                continue

            text_hash = hash_text(text)
            if text_hash is None:
                continue

            line = (
                f"{text_hash}\t{row.get('id', '')}\t"
                f"{str(row.get('author', '[deleted]')).replace(chr(10), ' ').replace(chr(9), ' ')}\t"
                f"{row.get('subreddit', 'unknown')}\t{str(row.get('date', ''))[:10]}"
            )
            lines.append(line)
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)

    return filepath.name, lines


def load_processed_files(logger):
    processed_files_file = STAGE2_CHECKPOINT_DIR / "processed_files.txt"
    if processed_files_file.exists():
        with open(processed_files_file, 'r') as f:
            processed = set(line.strip() for line in f)
        logger.info(f"  Loaded {len(processed):,} processed files from checkpoint")
        return processed
    return set()


def save_processed_files(processed_files):
    processed_files_file = STAGE2_CHECKPOINT_DIR / "processed_files.txt"
    with open(processed_files_file, 'w') as f:
        for fname in sorted(processed_files):
            f.write(f"{fname}\n")


def is_complete():
    flag_file = CHECKPOINTS_DIR / "stage2_complete.flag"
    return flag_file.exists()


def run(force=False):
    logger = setup_logging()
    STAGE2_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if is_complete() and not force:
        logger.info("Stage 2 already complete. Use force=True to rerun.")
        return True

    if force:
        for f in [TEMP_FILE, SORTED_FILE, STAGE2_CHECKPOINT_DIR / "processed_files.txt"]:
            if f.exists():
                f.unlink()

    logger.info("=" * 70)
    logger.info("STAGE 2: DUPLICATE ANALYSIS (Disk-based)")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    processed_files = load_processed_files(logger)

    all_files = get_all_csv_files()
    total_files = len(all_files)
    logger.info(f"\nFound {total_files} files total")

    remaining_files = [f for f in all_files if f.name not in processed_files]
    logger.info(f"Already processed: {len(processed_files):,} files")
    logger.info(f"Remaining to process: {len(remaining_files):,} files")

    if len(remaining_files) > 0:
        logger.info("\nPhase 1: Extracting hashes to disk...")

        with open(TEMP_FILE, 'a') as f_out:
            files_since_checkpoint = 0
            total_lines = 0

            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = {executor.submit(process_file, f): f for f in remaining_files}

                for i, future in enumerate(as_completed(futures)):
                    try:
                        filename, lines = future.result()

                        for line in lines:
                            f_out.write(line + '\n')
                        total_lines += len(lines)

                        processed_files.add(filename)
                        files_since_checkpoint += 1

                        total_processed = len(processed_files)
                        if total_processed % 500 == 0:
                            logger.info(f"  Processed {total_processed:,}/{total_files:,} files, {total_lines:,} rows written...")

                        if files_since_checkpoint >= CHECKPOINT_EVERY:
                            f_out.flush()
                            save_processed_files(processed_files)
                            logger.info(f"  Checkpoint saved at {total_processed:,} files")
                            files_since_checkpoint = 0

                    except Exception as e:
                        logger.error(f"Error: {e}")

            f_out.flush()
            save_processed_files(processed_files)

    result = subprocess.run(['wc', '-l', str(TEMP_FILE)], capture_output=True, text=True)
    total_items = int(result.stdout.split()[0])
    logger.info(f"\nTotal rows extracted: {total_items:,}")

    logger.info("\nPhase 2: Sorting by hash (split-sort-merge)...")

    split_dir = STAGE2_CHECKPOINT_DIR / "sort_chunks"
    split_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = 10
    lines_per_chunk = (total_items // num_chunks) + 1
    logger.info(f"  Splitting {total_items:,} lines into {num_chunks} chunks of ~{lines_per_chunk:,} lines...")

    split_cmd = f"split -l {lines_per_chunk} -d -a 2 {TEMP_FILE} {split_dir}/chunk_"
    result = subprocess.run(split_cmd, shell=True)
    if result.returncode != 0:
        logger.error("Split failed!")
        return False

    chunk_files = sorted(split_dir.glob("chunk_*"))
    logger.info(f"  Created {len(chunk_files)} chunks")

    sorted_chunks = []
    for i, chunk_file in enumerate(chunk_files):
        sorted_chunk = split_dir / f"sorted_{chunk_file.name}"
        sorted_chunks.append(sorted_chunk)

        if sorted_chunk.exists():
            logger.info(f"  Chunk {i+1}/{len(chunk_files)}: already sorted, skipping")
            continue

        logger.info(f"  Sorting chunk {i+1}/{len(chunk_files)}: {chunk_file.name}...")
        sort_cmd = f"sort -t'\t' -k1,1 -S 30G --parallel=8 {chunk_file} -o {sorted_chunk}"
        result = subprocess.run(sort_cmd, shell=True)
        if result.returncode != 0:
            logger.error(f"Sort failed on chunk {chunk_file.name}!")
            return False
        logger.info(f"  Chunk {i+1} sorted.")

    logger.info(f"  Merging {len(sorted_chunks)} sorted chunks...")
    sorted_chunk_args = " ".join(str(f) for f in sorted_chunks)
    merge_cmd = f"sort -t'\t' -k1,1 -m -S 4G {sorted_chunk_args} -o {SORTED_FILE}"
    result = subprocess.run(merge_cmd, shell=True)
    if result.returncode != 0:
        logger.error("Merge failed!")
        return False

    shutil.rmtree(split_dir)
    logger.info("  Sort complete.")

    logger.info("\nPhase 3: Analyzing duplicates...")

    stats = {
        'total_rows': total_items,
        'total_unique_hashes': 0,
        'duplicate_groups': 0,
        'total_duplicates': 0,
        'same_author_same_sub': 0,
        'same_author_diff_sub': 0,
        'diff_author_same_sub': 0,
        'diff_author_diff_sub': 0,
    }

    dup_details_file = OUTPUT_DIR / "duplicate_groups.jsonl"

    current_hash = None
    current_group = []

    def process_group(items, stats, f_ids, f_details):
        if len(items) < 2:
            return

        stats['duplicate_groups'] += 1
        stats['total_duplicates'] += len(items)

        authors = set(p['author'] for p in items)
        subreddits = set(p['subreddit'] for p in items)

        same_author = len(authors) == 1
        same_subreddit = len(subreddits) == 1

        if same_author and same_subreddit:
            pattern = 'same_author_same_sub'
        elif same_author and not same_subreddit:
            pattern = 'same_author_diff_sub'
        elif not same_author and same_subreddit:
            pattern = 'diff_author_same_sub'
        else:
            pattern = 'diff_author_diff_sub'

        stats[pattern] += 1

        sorted_items = sorted(items, key=lambda x: x['date'])

        for item in sorted_items[1:]:
            f_ids.write(f"{item['id']}\t{item['subreddit']}\t{items[0]['hash']}\n")

        group_info = {
            'hash': items[0]['hash'],
            'count': len(items),
            'pattern': pattern,
            'authors': list(authors),
            'subreddits': list(subreddits),
        }
        f_details.write(json.dumps(group_info) + '\n')

    with open(SORTED_FILE, 'r') as f_in, \
         open(DUPLICATE_IDS_FILE, 'w') as f_ids, \
         open(dup_details_file, 'w') as f_details:

        for line_num, line in enumerate(f_in):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            h, item_id, author, subreddit, date = parts[0], parts[1], parts[2], parts[3], parts[4]

            if h != current_hash:
                if current_group:
                    process_group(current_group, stats, f_ids, f_details)
                    stats['total_unique_hashes'] += 1

                current_hash = h
                current_group = []

            current_group.append({
                'hash': h,
                'id': item_id,
                'author': author,
                'subreddit': subreddit,
                'date': date
            })

            if (line_num + 1) % 10000000 == 0:
                logger.info(f"  Processed {line_num + 1:,} lines...")

        if current_group:
            process_group(current_group, stats, f_ids, f_details)
            stats['total_unique_hashes'] += 1

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total rows analyzed:         {stats['total_rows']:>12,}")
    logger.info(f"  Unique text contents:        {stats['total_unique_hashes']:>12,}")
    logger.info(f"  Duplicate groups:            {stats['duplicate_groups']:>12,}")
    logger.info(f"  Total duplicates:            {stats['total_duplicates']:>12,}")

    dup_rate = (stats['total_duplicates'] / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
    logger.info(f"  Duplication rate:            {dup_rate:>11.2f}%")

    logger.info(f"\nDuplicate Pattern Breakdown:")
    logger.info(f"  Same author, same subreddit:  {stats['same_author_same_sub']:>10,} groups")
    logger.info(f"  Same author, diff subreddits: {stats['same_author_diff_sub']:>10,} groups")
    logger.info(f"  Diff authors, same subreddit: {stats['diff_author_same_sub']:>10,} groups")
    logger.info(f"  Diff authors, diff subreddits:{stats['diff_author_diff_sub']:>10,} groups")

    stats_file = OUTPUT_DIR / "duplicate_analysis_stats.json"
    stats['timestamp'] = datetime.now().isoformat()
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nOutput files:")
    logger.info(f"  Stats: {stats_file}")
    logger.info(f"  Duplicate IDs: {DUPLICATE_IDS_FILE}")
    logger.info(f"  Duplicate groups: {dup_details_file}")

    flag_file = CHECKPOINTS_DIR / "stage2_complete.flag"
    flag_file.touch()

    logger.info("\n" + "=" * 70)
    logger.info(f"Stage 2 complete: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    return True


def main():
    global CHECKPOINT_EVERY

    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Duplicate Analysis (Disk-based)")
    parser.add_argument("--force", action="store_true", help="Force re-run from scratch")
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    args = parser.parse_args()

    CHECKPOINT_EVERY = args.checkpoint_interval

    success = run(force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
