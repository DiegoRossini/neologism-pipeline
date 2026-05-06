#!/usr/bin/env python3

import os
import json
import glob
import gzip
import logging
import heapq
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from config import (
    TEMP_BATCHES_DIR,
    CHECKPOINTS_DIR,
    LOG_DIR,
    TOKEN_COUNTS_FILE,
    UNIQUE_TOKENS_FILE,
)

FLUSH_EVERY = 50

def setup_logging():
    log_file = LOG_DIR / "stage_1_merge_batches.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_checkpoint():
    checkpoint_file = CHECKPOINTS_DIR / "stage1_merge_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'processed_batches': [], 'partial_files': []}

def save_checkpoint(processed_batches, partial_files):
    checkpoint_file = CHECKPOINTS_DIR / "stage1_merge_checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'processed_batches': processed_batches,
            'partial_files': partial_files
        }, f)

def is_complete():
    return TOKEN_COUNTS_FILE.exists() and UNIQUE_TOKENS_FILE.exists()

def mark_complete():
    checkpoint_file = CHECKPOINTS_DIR / "stage1_complete.flag"
    checkpoint_file.touch()

def flush_to_disk(token_counts, partial_num, partials_dir):
    """Write accumulated token counts to a sorted partial JSONL file."""
    partial_file = partials_dir / f"partial_{partial_num:04d}.jsonl"
    with open(partial_file, 'w') as f:
        for token in sorted(token_counts.keys()):
            f.write(json.dumps({
                'token': token,
                'counts': dict(token_counts[token])
            }) + '\n')
    logging.info(f"  Flushed {len(token_counts):,} tokens to {partial_file.name}")
    return str(partial_file)

def merge_partial_files(partial_files, output_file, unique_tokens_file):
    """Streaming merge of sorted partial JSONL files into final output.
    Each partial file is sorted by token, so we use a heap merge."""
    logging.info(f"Merging {len(partial_files)} partial files into {output_file}...")

    def token_iterator(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    yield entry['token'], entry['counts']

    iterators = [token_iterator(f) for f in partial_files]

    heap = []
    for i, it in enumerate(iterators):
        try:
            token, counts = next(it)
            heap.append((token, i, counts))
        except StopIteration:
            pass

    heapq.heapify(heap)

    unique_tokens = set()
    total_written = 0

    with open(output_file, 'w') as f_out:
        current_token = None
        current_counts = defaultdict(int)

        while heap:
            token, file_idx, counts = heapq.heappop(heap)

            try:
                next_token, next_counts = next(iterators[file_idx])
                heapq.heappush(heap, (next_token, file_idx, next_counts))
            except StopIteration:
                pass

            if token == current_token:
                for date, count in counts.items():
                    current_counts[date] += count
            else:
                if current_token is not None:
                    f_out.write(json.dumps({
                        'token': current_token,
                        'counts': dict(current_counts)
                    }) + '\n')
                    unique_tokens.add(current_token)
                    total_written += 1

                    if total_written % 5_000_000 == 0:
                        logging.info(f"  Written {total_written:,} tokens...")

                current_token = token
                current_counts = defaultdict(int)
                for date, count in counts.items():
                    current_counts[date] += count

        if current_token is not None:
            f_out.write(json.dumps({
                'token': current_token,
                'counts': dict(current_counts)
            }) + '\n')
            unique_tokens.add(current_token)
            total_written += 1

    logging.info(f"  Merged {total_written:,} unique tokens")

    logging.info(f"Writing unique tokens to {unique_tokens_file}...")
    with gzip.open(unique_tokens_file, 'wt', encoding='utf-8') as f:
        for token in tqdm(sorted(unique_tokens), desc="Writing unique tokens"):
            f.write(token + '\n')

    return total_written

def run(force=False):
    setup_logging()

    if is_complete() and not force:
        logging.info("Stage 1 already complete. Skipping. Use force=True to re-run.")
        return True

    if force:
        for f in [TOKEN_COUNTS_FILE, UNIQUE_TOKENS_FILE]:
            if f.exists():
                f.unlink()
        cp = CHECKPOINTS_DIR / "stage1_merge_checkpoint.json"
        if cp.exists():
            cp.unlink()
        flag = CHECKPOINTS_DIR / "stage1_complete.flag"
        if flag.exists():
            flag.unlink()

    logging.info("=" * 70)
    logging.info("STAGE 1: MERGING BATCH FILES (disk-based)")
    logging.info(f"Flush interval: every {FLUSH_EVERY} batches")
    logging.info("=" * 70)

    batch_files = sorted(glob.glob(str(TEMP_BATCHES_DIR / "batch_*.jsonl")))
    logging.info(f"Found {len(batch_files)} batch files to merge")

    if not batch_files:
        logging.error("No batch files found! Run stage 0 first.")
        return False

    partials_dir = Path(CHECKPOINTS_DIR) / "stage1_partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()
    processed_batches = set(checkpoint['processed_batches'])
    partial_files = list(checkpoint.get('partial_files', []))

    batches_to_process = [f for f in batch_files if os.path.basename(f) not in processed_batches]
    logging.info(f"Already processed: {len(processed_batches)} batches")
    logging.info(f"Existing partial files: {len(partial_files)}")
    logging.info(f"Remaining to process: {len(batches_to_process)} batches")

    if batches_to_process:
        token_counts = defaultdict(lambda: defaultdict(int))
        batches_since_flush = 0
        partial_num = len(partial_files)

        for batch_file in tqdm(batches_to_process, desc="Processing batches"):
            batch_name = os.path.basename(batch_file)

            try:
                with open(batch_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            token = entry['token']
                            for date, count in entry['counts'].items():
                                token_counts[token][date] += count

                processed_batches.add(batch_name)
                batches_since_flush += 1

                if batches_since_flush >= FLUSH_EVERY:
                    partial_num += 1
                    pf = flush_to_disk(token_counts, partial_num, partials_dir)
                    partial_files.append(pf)
                    save_checkpoint(list(processed_batches), partial_files)
                    logging.info(f"  Checkpoint: {len(processed_batches)} batches, {len(partial_files)} partials")
                    token_counts = defaultdict(lambda: defaultdict(int))
                    batches_since_flush = 0

                    import gc
                    gc.collect()

            except Exception as e:
                logging.error(f"Error processing {batch_name}: {e}")

        if token_counts:
            partial_num += 1
            pf = flush_to_disk(token_counts, partial_num, partials_dir)
            partial_files.append(pf)
            save_checkpoint(list(processed_batches), partial_files)
            del token_counts
            import gc
            gc.collect()

    logging.info(f"\nAll batches processed. {len(partial_files)} partial files to merge.")

    total_tokens = merge_partial_files(partial_files, TOKEN_COUNTS_FILE, UNIQUE_TOKENS_FILE)

    output_size_gb = TOKEN_COUNTS_FILE.stat().st_size / (1024**3)
    logging.info(f"Token counts file size: {output_size_gb:.2f} GB")

    unique_tokens_size_mb = UNIQUE_TOKENS_FILE.stat().st_size / (1024**2)
    logging.info(f"Unique tokens file size: {unique_tokens_size_mb:.2f} MB")

    logging.info("Cleaning up partial files...")
    import shutil
    shutil.rmtree(partials_dir, ignore_errors=True)

    mark_complete()

    logging.info("=" * 70)
    logging.info("STAGE 1 COMPLETE!")
    logging.info(f"Total unique tokens: {total_tokens:,}")
    logging.info(f"Outputs:")
    logging.info(f"  - {TOKEN_COUNTS_FILE}")
    logging.info(f"  - {UNIQUE_TOKENS_FILE}")
    logging.info("=" * 70)

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Merge Batches")
    parser.add_argument("--force", action="store_true", help="Force re-run even if complete")
    args = parser.parse_args()

    run(force=args.force)
