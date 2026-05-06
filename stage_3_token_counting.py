#!/usr/bin/env python3

import ast
import gc
import gzip
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm


def iter_chunks_safe(filepath, chunksize=500_000, usecols=None):
    try:
        chunks = pd.read_csv(
            filepath,
            compression='gzip',
            dtype=str,
            on_bad_lines='skip',
            chunksize=chunksize,
            usecols=usecols,
        )
        for chunk in chunks:
            yield chunk
        return
    except (UnicodeDecodeError, pd.errors.ParserError, EOFError, OSError) as e:
        logging.warning(f"  C engine failed on {filepath.name}: {e}. Falling back to Python engine.")

    try:
        chunks = pd.read_csv(
            gzip.open(filepath, 'rt', encoding='utf-8', errors='replace'),
            dtype=str,
            on_bad_lines='skip',
            chunksize=chunksize,
            engine='python',
            quoting=3,
        )
        for chunk in chunks:
            if usecols:
                available = [c for c in usecols if c in chunk.columns]
                if available:
                    chunk = chunk[available]
            yield chunk
    except Exception as e2:
        logging.error(f"  Python engine also failed on {filepath.name}: {e2}")
        return

from config import (
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    LOG_DIR,
    VOCAB_DIR,
    DUPLICATE_IDS_FILE,
    get_all_csv_files,
)

CHECKPOINT_EVERY = 500
DATASET_FREQ_DICT_MIN_COUNT = 2000
STAGE3_PARTIALS_DIR = CHECKPOINTS_DIR / "stage3_partials"
STAGE3_PROCESSED_FILE = CHECKPOINTS_DIR / "stage3_processed_files.txt"
STAGE3_DUP_PROCESSED_FILE = CHECKPOINTS_DIR / "stage3_dup_processed_files.txt"


def setup_logging():
    log_file = LOG_DIR / "stage_3_token_counting.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_tokenized_text(val):
    if pd.isna(val) or not isinstance(val, str):
        return []
    val = val.strip()
    if val.startswith('['):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


def load_processed_files(path):
    if path.exists():
        with open(path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_processed_files(processed, path):
    with open(path, 'w') as f:
        for fname in sorted(processed):
            f.write(fname + '\n')


def flush_partial(counter, partial_dir, prefix, partial_num):
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_file = partial_dir / f"{prefix}_{partial_num:04d}.tsv"
    with open(partial_file, 'w') as f:
        for token in sorted(counter.keys()):
            f.write(f"{token}\t{counter[token]}\n")
    logging.info(f"  Flushed {prefix} partial {partial_num}: {len(counter):,} tokens")


def merge_partials(partial_dir, prefix):
    files = sorted(partial_dir.glob(f"{prefix}_*.tsv"))
    if not files:
        return Counter()
    logging.info(f"  Merging {len(files)} {prefix} partial files...")
    merged = Counter()
    for f in files:
        with open(f, 'r') as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    merged[parts[0]] += int(parts[1])
    return merged


def is_complete():
    flag_file = CHECKPOINTS_DIR / "stage3_complete.flag"
    return flag_file.exists()


def pass1_raw_counts(all_files):
    logging.info("=" * 50)
    logging.info("PASS 1: Counting all tokens (raw, no dedup)")
    logging.info("=" * 50)

    processed_files = load_processed_files(STAGE3_PROCESSED_FILE)
    remaining = [f for f in all_files if f.name not in processed_files]
    logging.info(f"Already processed: {len(processed_files):,}, remaining: {len(remaining):,}")

    existing_partials = len(list(STAGE3_PARTIALS_DIR.glob("raw_*.tsv"))) if STAGE3_PARTIALS_DIR.exists() else 0
    partial_num = existing_partials

    token_freq = Counter()
    per_day = defaultdict(int)
    per_subreddit = {}
    total_rows = 0
    files_since_flush = 0

    for file_idx, filepath in enumerate(remaining):
        filename = filepath.name
        subreddit = filename.replace("_posts.csv.gz", "").replace("_text_comments.csv.gz", "").replace("_comments.csv.gz", "")

        try:
            chunks = iter_chunks_safe(filepath)

            file_tokens = set()
            for chunk in chunks:
                if 'tokenized_text' not in chunk.columns:
                    break

                valid = chunk['tokenized_text'].dropna()
                valid = valid[valid.str.startswith('[', na=False)]
                parsed = valid.apply(parse_tokenized_text)
                parsed = parsed[parsed.map(len) > 0]

                total_rows += len(parsed)

                for token_list in parsed:
                    tokens = set(token_list)
                    token_freq.update(tokens)
                    file_tokens.update(tokens)

                dates = chunk.loc[parsed.index, 'date'].fillna('').str[:10]
                date_counts = dates[dates != ''].value_counts()
                for date, count in date_counts.items():
                    if date != 'nan':
                        per_day[date] += count

                del chunk, valid, parsed, dates
                gc.collect()

            per_subreddit[subreddit] = len(file_tokens)

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

        processed_files.add(filename)
        files_since_flush += 1

        if (file_idx + 1) % 100 == 0:
            logging.info(f"  {len(processed_files):,}/{len(all_files)} files, {total_rows:,} rows, {len(token_freq):,} tokens")

        if files_since_flush >= CHECKPOINT_EVERY:
            partial_num += 1
            flush_partial(token_freq, STAGE3_PARTIALS_DIR, "raw", partial_num)

            meta_file = STAGE3_PARTIALS_DIR / f"meta_raw_{partial_num:04d}.json"
            with open(meta_file, 'w') as f:
                json.dump({'total_rows': total_rows, 'per_day': dict(per_day), 'per_subreddit': per_subreddit}, f)

            save_processed_files(processed_files, STAGE3_PROCESSED_FILE)
            logging.info(f"  Checkpoint: {len(processed_files):,} files, partial {partial_num}")

            token_freq = Counter()
            per_day = defaultdict(int)
            per_subreddit = {}
            total_rows = 0
            files_since_flush = 0
            gc.collect()

    if token_freq:
        partial_num += 1
        flush_partial(token_freq, STAGE3_PARTIALS_DIR, "raw", partial_num)
        meta_file = STAGE3_PARTIALS_DIR / f"meta_raw_{partial_num:04d}.json"
        with open(meta_file, 'w') as f:
            json.dump({'total_rows': total_rows, 'per_day': dict(per_day), 'per_subreddit': per_subreddit}, f)
        save_processed_files(processed_files, STAGE3_PROCESSED_FILE)
        del token_freq
        gc.collect()

    logging.info("Pass 1 complete.")
    return partial_num


DUP_IDS_NUMPY_FILE = CHECKPOINTS_DIR / "dup_ids_sorted.npy"


def build_numpy_dup_ids():
    import numpy as np

    if DUP_IDS_NUMPY_FILE.exists():
        logging.info(f"Loading pre-built numpy dedup array from {DUP_IDS_NUMPY_FILE}...")
        arr = np.load(DUP_IDS_NUMPY_FILE)
        logging.info(f"  Loaded {len(arr):,} IDs ({arr.nbytes / 1e9:.1f} GB)")
        return arr

    logging.info("Building sorted numpy array of duplicate IDs...")
    ids = []
    with open(DUPLICATE_IDS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0]:
                ids.append(parts[0])

    logging.info(f"  Read {len(ids):,} IDs, converting to numpy...")
    max_len = max(len(x) for x in ids)
    arr = np.array(ids, dtype=f'S{max_len}')
    del ids
    gc.collect()

    logging.info(f"  Sorting {len(arr):,} entries...")
    arr.sort()

    logging.info(f"  Saving to {DUP_IDS_NUMPY_FILE} ({arr.nbytes / 1e9:.1f} GB)...")
    np.save(DUP_IDS_NUMPY_FILE, arr)
    logging.info(f"  Done.")
    return arr


def is_dup_numpy(ids_series, sorted_arr):
    import numpy as np
    max_len = sorted_arr.dtype.itemsize
    query = ids_series.fillna('').astype(str).values
    query_bytes = np.array([x.encode('ascii', 'ignore')[:max_len] for x in query], dtype=sorted_arr.dtype)
    indices = np.searchsorted(sorted_arr, query_bytes)
    indices = np.clip(indices, 0, len(sorted_arr) - 1)
    return sorted_arr[indices] == query_bytes


def pass2_duplicate_counts(all_files):
    import numpy as np

    logging.info("=" * 50)
    logging.info("PASS 2: Counting tokens in duplicate rows (numpy)")
    logging.info("=" * 50)

    if not DUPLICATE_IDS_FILE.exists():
        logging.warning(f"No duplicate IDs file: {DUPLICATE_IDS_FILE}. Skipping pass 2.")
        return 0

    sorted_arr = build_numpy_dup_ids()

    processed_files = load_processed_files(STAGE3_DUP_PROCESSED_FILE)
    remaining = [f for f in all_files if f.name not in processed_files]
    logging.info(f"Already processed: {len(processed_files):,}, remaining: {len(remaining):,}")

    existing_partials = len(list(STAGE3_PARTIALS_DIR.glob("dup_*.tsv"))) if STAGE3_PARTIALS_DIR.exists() else 0
    partial_num = existing_partials

    dup_token_freq = Counter()
    dup_rows_total = 0
    files_since_flush = 0

    for file_idx, filepath in enumerate(remaining):
        try:
            for chunk in iter_chunks_safe(filepath, usecols=['id', 'tokenized_text']):
                if 'tokenized_text' not in chunk.columns or 'id' not in chunk.columns:
                    break

                is_dup = is_dup_numpy(chunk['id'], sorted_arr)
                dup_chunk = chunk[is_dup]

                if len(dup_chunk) == 0:
                    del chunk
                    continue

                valid = dup_chunk['tokenized_text'].dropna()
                valid = valid[valid.str.startswith('[', na=False)]
                parsed = valid.apply(parse_tokenized_text)
                parsed = parsed[parsed.map(len) > 0]

                dup_rows_total += len(parsed)
                for token_list in parsed:
                    dup_token_freq.update(set(token_list))

                del chunk, dup_chunk
                gc.collect()

        except Exception as e:
            logging.error(f"Error processing {filepath.name}: {e}")

        processed_files.add(filepath.name)
        files_since_flush += 1

        if (file_idx + 1) % 50 == 0:
            logging.info(
                f"  {file_idx + 1:,}/{len(remaining)} files, "
                f"{dup_rows_total:,} dup rows, {len(dup_token_freq):,} tokens"
            )

        if files_since_flush >= CHECKPOINT_EVERY:
            partial_num += 1
            flush_partial(dup_token_freq, STAGE3_PARTIALS_DIR, "dup", partial_num)

            meta_file = STAGE3_PARTIALS_DIR / f"meta_dup_{partial_num:04d}.json"
            with open(meta_file, 'w') as f_meta:
                json.dump({'dup_rows': dup_rows_total}, f_meta)

            save_processed_files(processed_files, STAGE3_DUP_PROCESSED_FILE)
            logging.info(f"  Checkpoint: {len(processed_files):,} files, partial {partial_num}")

            dup_token_freq = Counter()
            dup_rows_total = 0
            files_since_flush = 0

    if dup_token_freq:
        partial_num += 1
        flush_partial(dup_token_freq, STAGE3_PARTIALS_DIR, "dup", partial_num)
        meta_file = STAGE3_PARTIALS_DIR / f"meta_dup_{partial_num:04d}.json"
        with open(meta_file, 'w') as f_meta:
            json.dump({'dup_rows': dup_rows_total}, f_meta)
        save_processed_files(processed_files, STAGE3_DUP_PROCESSED_FILE)

    del sorted_arr
    gc.collect()

    logging.info("Pass 2 complete.")
    return partial_num


def pass3_merge_and_write():
    logging.info("=" * 50)
    logging.info("PASS 3: Merging and writing final outputs")
    logging.info("=" * 50)

    token_freq_raw = merge_partials(STAGE3_PARTIALS_DIR, "raw")
    logging.info(f"  Raw tokens: {len(token_freq_raw):,}")

    dup_freq = merge_partials(STAGE3_PARTIALS_DIR, "dup")
    logging.info(f"  Duplicate tokens: {len(dup_freq):,}")

    total_rows = 0
    total_dup_rows = 0
    per_day_raw = defaultdict(int)
    per_subreddit_all = {}

    for meta_file in sorted(STAGE3_PARTIALS_DIR.glob("meta_raw_*.json")):
        with open(meta_file) as f:
            data = json.load(f)
        total_rows += data['total_rows']
        for date, count in data['per_day'].items():
            per_day_raw[date] += count
        per_subreddit_all.update(data['per_subreddit'])

    for meta_file in sorted(STAGE3_PARTIALS_DIR.glob("meta_dup_*.json")):
        with open(meta_file) as f:
            data = json.load(f)
        total_dup_rows += data['dup_rows']

    total_rows_dedup = total_rows - total_dup_rows

    token_freq_dedup = Counter()
    for token, raw_count in token_freq_raw.items():
        dedup_count = raw_count - dup_freq.get(token, 0)
        if dedup_count > 0:
            token_freq_dedup[token] = dedup_count

    logging.info(f"  Deduped tokens: {len(token_freq_dedup):,}")

    logging.info("\nWriting output files...")

    freq_file = OUTPUT_DIR / "token_frequencies.tsv"
    with open(freq_file, 'w') as f:
        f.write("token\tfrequency\tfrequency_deduped\n")
        for token in sorted(token_freq_raw.keys()):
            raw = token_freq_raw[token]
            dedup = token_freq_dedup.get(token, 0)
            f.write(f"{token}\t{raw}\t{dedup}\n")
    logging.info(f"  Written {len(token_freq_raw):,} tokens to {freq_file}")

    day_file = OUTPUT_DIR / "unique_tokens_per_day.tsv"
    with open(day_file, 'w') as f:
        f.write("date\trows\n")
        for date in sorted(per_day_raw.keys()):
            f.write(f"{date}\t{per_day_raw[date]}\n")
    logging.info(f"  Written {len(per_day_raw)} dates to {day_file}")

    sub_file = OUTPUT_DIR / "unique_tokens_per_subreddit.tsv"
    with open(sub_file, 'w') as f:
        f.write("subreddit\tunique_tokens\n")
        for sub in sorted(per_subreddit_all.keys()):
            f.write(f"{sub}\t{per_subreddit_all[sub]}\n")
    logging.info(f"  Written {len(per_subreddit_all)} subreddits to {sub_file}")

    dict_path = VOCAB_DIR / "dataset_frequency_dict_deduped.txt"
    dict_count = 0
    with open(dict_path, 'w', encoding='utf-8') as f:
        for token, count in token_freq_dedup.most_common():
            if count >= DATASET_FREQ_DICT_MIN_COUNT and token.isalpha():
                f.write(f"{token}\t{count}\n")
                dict_count += 1
    logging.info(f"  Written {dict_count:,} tokens to {dict_path} (deduped freq >= {DATASET_FREQ_DICT_MIN_COUNT:,})")

    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': total_rows,
        'total_rows_deduped': total_rows_dedup,
        'total_duplicate_rows': total_dup_rows,
        'total_unique_tokens': len(token_freq_raw),
        'total_unique_tokens_deduped': len(token_freq_dedup),
        'num_dates': len(per_day_raw),
        'num_subreddits': len(per_subreddit_all),
        'dataset_freq_dict_terms': dict_count,
    }
    stats_file = OUTPUT_DIR / "token_counting_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"  Stats written to {stats_file}")

    logging.info("\n" + "=" * 70)
    logging.info("RESULTS")
    logging.info("=" * 70)
    logger = logging.getLogger(__name__)
    logger.info(f"  Total rows processed:          {stats['total_rows']:>12,}")
    logger.info(f"  Total duplicate rows:          {stats['total_duplicate_rows']:>12,}")
    logger.info(f"  Total rows after dedup:        {stats['total_rows_deduped']:>12,}")
    logger.info(f"  Total unique tokens:           {stats['total_unique_tokens']:>12,}")
    logger.info(f"  Total unique tokens (deduped): {stats['total_unique_tokens_deduped']:>12,}")
    logger.info(f"  Number of dates:               {stats['num_dates']:>12,}")
    logger.info(f"  Number of subreddits:          {stats['num_subreddits']:>12,}")

    logging.info("Cleaning up partial files...")
    import shutil
    shutil.rmtree(STAGE3_PARTIALS_DIR, ignore_errors=True)
    for f in [STAGE3_PROCESSED_FILE, STAGE3_DUP_PROCESSED_FILE]:
        if f.exists():
            f.unlink()

    return stats


def run(force=False):
    logger = setup_logging()

    if is_complete() and not force:
        logger.info("Stage 3 already complete. Use force=True to rerun.")
        return True

    if force:
        flag = CHECKPOINTS_DIR / "stage3_complete.flag"
        if flag.exists():
            flag.unlink()
        if STAGE3_PROCESSED_FILE.exists():
            STAGE3_PROCESSED_FILE.unlink()
        if STAGE3_DUP_PROCESSED_FILE.exists():
            STAGE3_DUP_PROCESSED_FILE.unlink()
        import shutil
        if STAGE3_PARTIALS_DIR.exists():
            shutil.rmtree(STAGE3_PARTIALS_DIR)

    logger.info("=" * 70)
    logger.info("STAGE 3: TOKEN COUNTING (3-pass: raw, duplicates, subtract)")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    all_files = get_all_csv_files()
    logger.info(f"Found {len(all_files)} files to process")

    pass1_raw_counts(all_files)
    pass2_duplicate_counts(all_files)
    stats = pass3_merge_and_write()

    flag_file = CHECKPOINTS_DIR / "stage3_complete.flag"
    flag_file.touch()

    logger.info("\n" + "=" * 70)
    logger.info(f"Stage 3 complete: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Token Counting")
    parser.add_argument("--force", action="store_true", help="Force re-run from scratch")
    args = parser.parse_args()

    success = run(force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
