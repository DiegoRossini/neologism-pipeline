#!/usr/bin/env python3

import gc
import glob
import gzip
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from config import (
    DATA_DIRS,
    TEMP_BATCHES_DIR,
    CHECKPOINTS_DIR,
    LOG_DIR,
    TOKENIZATION_CONFIG,
    TEST_MODE as GLOBAL_TEST_MODE,
    TEST_FILES_LIMIT,
    detect_text_column,
    get_all_csv_files,
)

N_CORES = TOKENIZATION_CONFIG["n_cores"]
BATCH_SIZE = TOKENIZATION_CONFIG["batch_size"]
TEST_MODE = GLOBAL_TEST_MODE
TEST_FILES = TEST_FILES_LIMIT

def setup_logging():
    log_file = LOG_DIR / "stage_0_tokenization.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HASHTAG_PATTERN = re.compile(r'#\w+')
SUBREDDIT_PATTERN = re.compile(r'r/\w+')
MENTION_PATTERN = re.compile(r'u/\w+')
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def init_tokenizer():
    nlp = spacy.load("en_core_web_lg", disable=["ner", "parser", "lemmatizer"])
    return nlp

def preprocess_text(text):
    text = URL_PATTERN.sub(' URL ', text)
    text = SUBREDDIT_PATTERN.sub(' SUBREDDIT ', text)
    text = MENTION_PATTERN.sub(' USER ', text)
    text = HASHTAG_PATTERN.sub(' HASHTAG_TOKEN ', text)
    text = EMOJI_PATTERN.sub(' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(text.split())
    return text

def tokenize_text(nlp, text):
    text = preprocess_text(text)
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct:
            continue
        if token.text.lower() in STOP_WORDS:
            continue
        if token.is_space:
            continue
        tokens.append(token.text.lower())
    return tokens

def tokenize_chunk(chunk_data):
    texts, chunk_id = chunk_data
    nlp = init_tokenizer()
    results = []
    for text in texts:
        if pd.notna(text):
            try:
                tokens = tokenize_text(nlp, str(text))
            except Exception:
                tokens = []
        else:
            tokens = []
        results.append(tokens)
    return chunk_id, results

def load_checkpoint():
    checkpoint_file = CHECKPOINTS_DIR / "stage0_processed_files.txt"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_checkpoint(processed_file):
    checkpoint_file = CHECKPOINTS_DIR / "stage0_processed_files.txt"
    with open(checkpoint_file, 'a') as f:
        f.write(f"{processed_file}\n")

def get_file_size_gb(file_path):
    return os.path.getsize(file_path) / (1024**3)

LARGE_FILE_THRESHOLD_GB = 0.5
CHUNK_ROWS = 1_000_000

def process_large_file_chunked(file_path, text_col):
    try:
        filename = os.path.basename(file_path)
        file_size_gb = get_file_size_gb(file_path)

        logging.info(f"Processing {filename} ({file_size_gb:.2f} GB) in CHUNKED mode...")
        start_time = time.time()

        n_cores_to_use = min(N_CORES, 16)

        token_date_counts = defaultdict(lambda: defaultdict(int))
        total_rows = 0
        chunk_num = 0

        temp_output = f"{file_path}.chunked_tmp"
        first_chunk = True

        with gzip.open(file_path, 'rt') as f:
            header = f.readline().strip().split(',')
        has_tokenized = 'tokenized_text' in header

        if has_tokenized:
            logging.info(f"  - File already has tokenized_text column, extracting counts only (chunked)")
        else:
            logging.info(f"  - Will tokenize with {n_cores_to_use} cores (chunked, {CHUNK_ROWS:,} rows/chunk)")

        with gzip.open(temp_output, 'wt') as out_f:
            for chunk in pd.read_csv(file_path, compression='gzip',
                                      dtype={text_col: str, 'date': str},
                                      on_bad_lines='skip',
                                      chunksize=CHUNK_ROWS):
                chunk_num += 1
                chunk_start = time.time()
                logging.info(f"  - Processing chunk {chunk_num} ({len(chunk):,} rows)...")

                if has_tokenized:
                    if chunk['tokenized_text'].dtype in ('object', 'str', 'string'):
                        chunk['tokenized_text'] = chunk['tokenized_text'].apply(
                            lambda x: eval(x) if pd.notna(x) and isinstance(x, str) and x.startswith('[') else []
                        )
                else:
                    chunk_size_inner = max(1000, len(chunk) // (n_cores_to_use * 2))
                    chunks_to_process = []

                    for i in range(0, len(chunk), chunk_size_inner):
                        end_idx = min(i + chunk_size_inner, len(chunk))
                        chunk_texts = chunk[text_col].iloc[i:end_idx].tolist()
                        chunks_to_process.append((chunk_texts, len(chunks_to_process)))

                    with Pool(processes=n_cores_to_use) as pool:
                        results = list(pool.imap(tokenize_chunk, chunks_to_process))

                    results.sort(key=lambda x: x[0])
                    tokenized = []
                    for _, chunk_results in results:
                        tokenized.extend(chunk_results)

                    chunk['tokenized_text'] = tokenized

                for _, row in chunk.iterrows():
                    if pd.notna(row['date']) and row['tokenized_text']:
                        date = str(row['date'])[:10]
                        for token in row['tokenized_text']:
                            token_date_counts[token][date] += 1

                chunk['tokenized_text'] = chunk['tokenized_text'].apply(str)
                chunk.to_csv(out_f, index=False, header=first_chunk)
                first_chunk = False

                total_rows += len(chunk)
                chunk_time = time.time() - chunk_start
                logging.info(f"    Chunk {chunk_num} done in {chunk_time:.1f}s (total: {total_rows:,} rows)")

                del chunk
                gc.collect()

        logging.info(f"  - Compressing output file...")
        final_temp = f"{file_path}.tmp"
        with open(temp_output, 'rb') as f_in:
            with gzip.open(final_temp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(temp_output)
        shutil.move(final_temp, file_path)
        logging.info(f"  - File saved successfully")

        total_time = time.time() - start_time
        logging.info(f"  ✓ Completed {filename}: {total_rows:,} texts, {len(token_date_counts):,} unique tokens in {total_time:.1f}s")

        return token_date_counts, {'file': filename, 'total_texts': total_rows, 'unique_tokens': len(token_date_counts)}

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        for temp in [f"{file_path}.chunked_tmp", f"{file_path}.tmp"]:
            if os.path.exists(temp):
                os.remove(temp)
        return None, None

def process_file_inplace(file_path):
    if not os.path.exists(file_path):
        logging.warning(f"File not found, skipping: {file_path}")
        return None, None

    text_col = detect_text_column(file_path)
    if text_col is None:
        logging.warning(f"No recognized text column in {file_path}, skipping")
        return None, None

    file_size_gb = get_file_size_gb(file_path)
    if file_size_gb > LARGE_FILE_THRESHOLD_GB:
        return process_large_file_chunked(file_path, text_col)

    try:
        filename = os.path.basename(file_path)

        logging.info(f"Processing {filename} ({file_size_gb:.2f} GB)...")
        start_time = time.time()

        logging.info(f"  - Loading file...")
        df = pd.read_csv(file_path, compression='gzip', dtype={text_col: str, 'date': str}, on_bad_lines='skip')
        logging.info(f"  - Loaded {len(df):,} rows")

        if 'tokenized_text' in df.columns:
            logging.info(f"  - File already has tokenized_text column, extracting counts only")
            if df['tokenized_text'].dtype in ('object', 'str', 'string') and len(df) > 0:
                sample = df['tokenized_text'].dropna().iloc[0] if df['tokenized_text'].notna().any() else None
                if sample is not None and isinstance(sample, str) and sample.startswith('['):
                    df['tokenized_text'] = df['tokenized_text'].apply(
                        lambda x: eval(x) if pd.notna(x) and isinstance(x, str) else []
                    )
        else:
            n_cores_to_use = min(N_CORES, 32)

            logging.info(f"  - Starting parallel tokenization with {n_cores_to_use} cores...")

            chunk_size = max(1000, len(df) // (n_cores_to_use * 4))
            chunks = []

            for i in range(0, len(df), chunk_size):
                end_idx = min(i + chunk_size, len(df))
                chunk_texts = df[text_col].iloc[i:end_idx].tolist()
                chunks.append((chunk_texts, len(chunks)))

            logging.info(f"  - Split into {len(chunks)} chunks of ~{chunk_size} rows each")

            with Pool(processes=n_cores_to_use) as pool:
                results = list(tqdm(
                    pool.imap(tokenize_chunk, chunks),
                    total=len(chunks),
                    desc=f"  - Tokenizing {filename}",
                    unit="chunks",
                    leave=True,
                    file=sys.stderr
                ))

            results.sort(key=lambda x: x[0])
            tokenized = []
            for chunk_id, chunk_results in results:
                tokenized.extend(chunk_results)

            df['tokenized_text'] = tokenized

            elapsed = time.time() - start_time
            logging.info(f"  - Tokenization complete in {elapsed:.1f} seconds")

        logging.info(f"  - Saving file with tokenized column...")
        temp_path = f"{file_path}.tmp"
        df.to_csv(temp_path, compression='gzip', index=False)
        shutil.move(temp_path, file_path)
        logging.info(f"  - File saved successfully")

        logging.info(f"  - Extracting token-date counts...")
        token_date_counts = defaultdict(lambda: defaultdict(int))

        for idx, row in df.iterrows():
            if pd.notna(row['date']) and row['tokenized_text']:
                date = str(row['date'])[:10]
                for token in row['tokenized_text']:
                    token_date_counts[token][date] += 1

        metadata = {
            'file': filename,
            'total_texts': len(df),
            'unique_authors': df['author'].nunique() if 'author' in df.columns else 0,
            'unique_tokens': len(token_date_counts),
        }

        total_time = time.time() - start_time
        logging.info(f"  ✓ Completed {filename}: {len(df):,} texts, {len(token_date_counts):,} unique tokens in {total_time:.1f}s")

        return token_date_counts, metadata

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        temp_path = f"{file_path}.tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None

def save_batch(token_accumulator, batch_num):
    batch_file = TEMP_BATCHES_DIR / f"batch_{batch_num:04d}.jsonl"

    with open(batch_file, 'w') as f:
        for token, date_counts in token_accumulator.items():
            json_line = json.dumps({
                'token': token,
                'counts': dict(date_counts)
            })
            f.write(json_line + '\n')

    logging.info(f"Saved batch {batch_num} with {len(token_accumulator):,} tokens")

def merge_token_counts(accumulator, new_counts):
    for token, date_counts in new_counts.items():
        for date, count in date_counts.items():
            accumulator[token][date] += count

def is_complete():
    batch_files = list(TEMP_BATCHES_DIR.glob("batch_*.jsonl"))
    checkpoint_file = CHECKPOINTS_DIR / "stage0_complete.flag"
    return checkpoint_file.exists() and len(batch_files) > 0

def mark_complete():
    checkpoint_file = CHECKPOINTS_DIR / "stage0_complete.flag"
    checkpoint_file.touch()

def run(force=False):
    setup_logging()

    if is_complete() and not force:
        logging.info("Stage 0 already complete. Skipping. Use force=True to re-run.")
        return True

    logging.info("=" * 70)
    logging.info(f"STAGE 0: TOKENIZATION - Using {N_CORES} CPU cores")
    logging.info(f"Data directories: {[str(d) for d in DATA_DIRS]}")
    logging.info(f"Output batch directory: {TEMP_BATCHES_DIR}")
    logging.info("=" * 70)

    all_files = [str(f) for f in get_all_csv_files()]
    logging.info(f"Found {len(all_files)} total files")

    processed_files = load_checkpoint()
    files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]

    if TEST_MODE:
        files_to_process = files_to_process[:TEST_FILES]
        logging.info(f"TEST MODE: Processing only {TEST_FILES} files")
    else:
        logging.info(f"FULL MODE: {len(files_to_process)} files remaining to process")

    if not files_to_process:
        logging.info("No files to process. All done!")
        mark_complete()
        return True

    total_gb = sum(get_file_size_gb(f) for f in files_to_process)
    logging.info(f"Total data to process: {total_gb:.2f} GB")

    token_accumulator = defaultdict(lambda: defaultdict(int))
    batch_num = len(processed_files) // BATCH_SIZE

    for i, file_path in enumerate(files_to_process):
        logging.info(f"\n[{i+1}/{len(files_to_process)}] Processing next file...")

        token_counts, metadata = process_file_inplace(file_path)

        if token_counts is None:
            logging.warning(f"Skipping {file_path} due to error")
            continue

        merge_token_counts(token_accumulator, token_counts)
        save_checkpoint(os.path.basename(file_path))

        del token_counts
        gc.collect()

        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(files_to_process):
            batch_num += 1
            save_batch(token_accumulator, batch_num)
            token_accumulator = defaultdict(lambda: defaultdict(int))
            gc.collect()

            logging.info(f"Overall progress: {i+1}/{len(files_to_process)} files completed")

    mark_complete()
    logging.info("=" * 70)
    logging.info("STAGE 0 COMPLETE!")
    logging.info("=" * 70)

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 0: Tokenization")
    parser.add_argument("--force", action="store_true", help="Force re-run even if complete")
    parser.add_argument("--test", type=int, metavar="N", help="Test mode: process only N files")
    args = parser.parse_args()

    if args.test:
        TEST_MODE = True
        TEST_FILES = args.test

    run(force=args.force)
