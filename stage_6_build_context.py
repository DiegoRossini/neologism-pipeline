#!/usr/bin/env python3

import ast
import gzip
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    CHECKPOINTS_DIR,
    LOG_DIR,
    CANDIDATE_NEOLOGISMS_FILE,
    CONTEXT_INDEX_FILE,
    CONTEXT_CONFIG,
    DUPLICATE_IDS_FILE,
    TEST_MODE,
    TEST_FILES_LIMIT,
    detect_text_column,
    get_all_csv_files,
)

DUP_IDS_NUMPY_FILE = CHECKPOINTS_DIR / "dup_ids_sorted.npy"

CHECKPOINT_EVERY = 500

_SHARED_TARGET_TOKENS = None
_SHARED_DUP_IDS_ARR = None
_SHARED_MAX_CONTEXTS = None
_SHARED_MAX_TOKEN_COUNT = None

def _init_worker(target_tokens, dup_ids_arr, max_contexts, max_token_count):
    global _SHARED_TARGET_TOKENS, _SHARED_DUP_IDS_ARR
    global _SHARED_MAX_CONTEXTS, _SHARED_MAX_TOKEN_COUNT
    _SHARED_TARGET_TOKENS = target_tokens
    _SHARED_DUP_IDS_ARR = dup_ids_arr
    _SHARED_MAX_CONTEXTS = max_contexts
    _SHARED_MAX_TOKEN_COUNT = max_token_count


def is_dup_numpy(ids_series, sorted_arr):
    if sorted_arr is None or len(sorted_arr) == 0:
        return np.zeros(len(ids_series), dtype=bool)
    max_len = sorted_arr.dtype.itemsize
    query = ids_series.fillna('').astype(str).values
    query_bytes = np.array([x.encode('ascii', 'ignore')[:max_len] for x in query], dtype=sorted_arr.dtype)
    indices = np.searchsorted(sorted_arr, query_bytes)
    indices = np.clip(indices, 0, len(sorted_arr) - 1)
    return sorted_arr[indices] == query_bytes

def setup_logging():
    log_file = LOG_DIR / "stage_6_build_context.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stream_handler]
    )

def load_candidate_tokens(filepath):
    tokens = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.add(token)
    return tokens

def load_duplicate_ids_numpy():
    if not DUPLICATE_IDS_FILE.exists():
        logging.warning(f"Duplicate IDs file not found: {DUPLICATE_IDS_FILE}")
        logging.warning("Proceeding without deduplication")
        return None

    if DUP_IDS_NUMPY_FILE.exists():
        logging.info(f"Loading pre-built numpy dedup array from {DUP_IDS_NUMPY_FILE}...")
        arr = np.load(DUP_IDS_NUMPY_FILE)
        logging.info(f"  Loaded {len(arr):,} IDs ({arr.nbytes / 1e9:.1f} GB)")
        return arr

    logging.info(f"Building sorted numpy array from {DUPLICATE_IDS_FILE}...")
    ids = []
    with open(DUPLICATE_IDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0]:
                ids.append(parts[0])
    logging.info(f"  Read {len(ids):,} IDs, converting to numpy...")
    max_len = max(len(x) for x in ids) if ids else 1
    arr = np.array(ids, dtype=f'S{max_len}')
    del ids
    import gc
    gc.collect()
    logging.info(f"  Sorting {len(arr):,} entries...")
    arr.sort()
    logging.info(f"  Saving to {DUP_IDS_NUMPY_FILE} ({arr.nbytes / 1e9:.1f} GB)...")
    np.save(DUP_IDS_NUMPY_FILE, arr)
    return arr

def process_single_file(filepath):
    target_tokens = _SHARED_TARGET_TOKENS
    dup_ids_arr = _SHARED_DUP_IDS_ARR
    max_contexts_per_token = _SHARED_MAX_CONTEXTS
    max_token_count = _SHARED_MAX_TOKEN_COUNT

    contexts = defaultdict(list)

    try:
        text_col = detect_text_column(filepath)
        if text_col is None:
            return dict(contexts)
        cols_to_load = ['tokenized_text', 'subreddit', 'id', text_col]
        try:
            df = pd.read_csv(filepath, compression='gzip', usecols=cols_to_load, on_bad_lines='skip')
        except (UnicodeDecodeError, pd.errors.ParserError, EOFError, OSError):
            try:
                df = pd.read_csv(
                    gzip.open(filepath, 'rt', encoding='utf-8', errors='replace'),
                    on_bad_lines='skip', engine='python', quoting=3,
                )
                avail = [c for c in cols_to_load if c in df.columns]
                if not avail or 'tokenized_text' not in avail:
                    return dict(contexts)
                df = df[avail]
            except Exception:
                return dict(contexts)

        if dup_ids_arr is not None and 'id' in df.columns:
            is_dup = is_dup_numpy(df['id'], dup_ids_arr)
            df = df[~is_dup]

        for _, row in df.iterrows():
            tokenized_text = row.get('tokenized_text')
            original_text = row.get(text_col)
            subreddit = str(row.get('subreddit', 'unknown'))
            post_id = str(row.get('id', ''))

            if pd.isna(tokenized_text) or pd.isna(original_text):
                continue

            try:
                tokens_in_text = ast.literal_eval(tokenized_text)
                if not isinstance(tokens_in_text, list):
                    continue
            except (ValueError, SyntaxError):
                continue

            tokens_in_text_set = set(tokens_in_text)
            matching_tokens = target_tokens & tokens_in_text_set

            token_count = len(tokens_in_text)
            text_str = str(original_text)
            text_lower = text_str.lower()
            text_len = len(text_str)

            def context_richness(tok):
                pos = text_lower.find(tok.lower())
                if pos == -1:
                    return -1
                return min(pos, text_len - pos - len(tok))

            sorted_tokens = sorted(matching_tokens, key=context_richness, reverse=True)

            for token in sorted_tokens:
                if len(contexts[token]) < max_contexts_per_token:
                    text = str(original_text)

                    text_lower = text.lower()
                    token_lower = token.lower()
                    token_pos = text_lower.find(token_lower)
                    if token_pos == -1:
                        continue

                    if token_count > max_token_count:
                        words = text.split()
                        char_count = 0
                        token_word_idx = 0
                        for i, w in enumerate(words):
                            if i > 0:
                                char_count += 1
                            if char_count + len(w) > token_pos:
                                token_word_idx = i
                                break
                            char_count += len(w)

                        half_window = max_token_count // 2
                        start = max(0, token_word_idx - half_window)
                        end = start + max_token_count
                        if end > len(words):
                            end = len(words)
                            start = max(0, end - max_token_count)

                        selected_words = words[start:end]
                        text = " ".join(selected_words)
                        if start > 0:
                            text = "... " + text
                        if end < len(words):
                            text = text + " ..."
                        token_count = len(selected_words)

                    if token.lower() not in text.lower():
                        continue

                    contexts[token].append((text, token_count, subreddit, post_id))

    except Exception:
        pass

    return dict(contexts)

def merge_contexts(all_contexts, max_contexts, min_token_count):
    merged = defaultdict(list)

    for contexts in all_contexts:
        for token, tuples in contexts.items():
            merged[token].extend(tuples)

    final = {}
    for token, tuples in merged.items():
        pool = sorted(tuples, key=lambda x: x[1], reverse=True)
        good = [t for t in pool if t[1] >= min_token_count]
        if good:
            pool = good

        by_subreddit = defaultdict(list)
        for t in pool:
            by_subreddit[t[2]].append(t)

        sub_counts = sorted(by_subreddit.items(), key=lambda x: -len(x[1]))

        if len(sub_counts) <= 1:
            selected = pool[:max_contexts]
        else:
            n_subs = min(len(sub_counts), 3)
            slots = {}
            per_sub = max_contexts // n_subs
            remainder = max_contexts % n_subs
            for i in range(n_subs):
                sub_name = sub_counts[i][0]
                slots[sub_name] = per_sub + (1 if i < remainder else 0)

            selected = []
            for sub_name, n_slots in slots.items():
                sub_contexts = by_subreddit[sub_name]
                selected.extend(sub_contexts[:n_slots])

            if len(selected) < max_contexts:
                selected_set = set(id(t) for t in selected)
                for sub_name, sub_contexts in sub_counts:
                    if sub_name in slots:
                        continue
                    for t in sub_contexts:
                        if id(t) not in selected_set:
                            selected.append(t)
                            if len(selected) >= max_contexts:
                                break
                    if len(selected) >= max_contexts:
                        break

        final[token] = [
            {"text": t[0], "subreddit": t[2], "post_id": t[3]}
            for t in selected
        ]

    return final

STAGE6_PARTIALS_DIR = CHECKPOINTS_DIR / "stage6_partials"


def flush_contexts_to_disk(contexts_buffer, partial_num):
    STAGE6_PARTIALS_DIR.mkdir(parents=True, exist_ok=True)
    partial_file = STAGE6_PARTIALS_DIR / f"contexts_{partial_num:05d}.jsonl"
    with open(partial_file, 'w', encoding='utf-8') as f:
        for contexts in contexts_buffer:
            for token, tuples in contexts.items():
                for t in tuples:
                    f.write(json.dumps({
                        "token": token,
                        "text": t[0],
                        "token_count": t[1],
                        "subreddit": t[2],
                        "post_id": t[3],
                    }, ensure_ascii=False) + '\n')


def save_checkpoint(processed_files, tokens_found, partial_num, checkpoint_dir):
    checkpoint_file = checkpoint_dir / "stage6_checkpoint.json"
    checkpoint_data = {
        "processed_files": processed_files,
        "tokens_found": list(tokens_found),
        "partial_num": partial_num,
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    logging.info(f"[Checkpoint] Saved: {len(processed_files)} files, {len(tokens_found)} tokens, partial {partial_num}")


def load_checkpoint(checkpoint_dir):
    checkpoint_file = checkpoint_dir / "stage6_checkpoint.json"
    if not checkpoint_file.exists():
        return None
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        processed_files = set(checkpoint_data["processed_files"])
        tokens_found = set(checkpoint_data["tokens_found"])
        partial_num = checkpoint_data.get("partial_num", 0)
        logging.info(f"[Checkpoint] Resumed: {len(processed_files)} files already processed, {len(tokens_found)} tokens found, partial {partial_num}")
        return processed_files, tokens_found, partial_num
    except Exception as e:
        logging.warning(f"[Checkpoint] Failed to load: {e}, starting fresh")
        return None


def load_all_partials():
    if not STAGE6_PARTIALS_DIR.exists():
        return []
    partial_files = sorted(STAGE6_PARTIALS_DIR.glob("contexts_*.jsonl"))
    logging.info(f"Loading {len(partial_files)} partial files for merge...")
    all_contexts = [defaultdict(list)]
    for pf in partial_files:
        with open(pf, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                all_contexts[0][entry["token"]].append(
                    (entry["text"], entry["token_count"], entry["subreddit"], entry["post_id"])
                )
    return all_contexts


def clean_checkpoint(checkpoint_dir):
    import shutil
    cp_file = checkpoint_dir / "stage6_checkpoint.json"
    if cp_file.exists():
        cp_file.unlink()
    if STAGE6_PARTIALS_DIR.exists():
        shutil.rmtree(STAGE6_PARTIALS_DIR, ignore_errors=True)
    logging.info("[Checkpoint] Cleaned up checkpoint files")

def is_complete():
    return CONTEXT_INDEX_FILE.exists()

def mark_complete():
    checkpoint_file = CHECKPOINTS_DIR / "stage6_context_complete.flag"
    checkpoint_file.touch()

def run(force=False):
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=False)
    except RuntimeError:
        pass

    setup_logging()

    if is_complete() and not force:
        logging.info("Stage 6 already complete. Skipping. Use force=True to re-run.")
        return True

    logging.info("=" * 70)
    logging.info("STAGE 6: BUILD CONTEXT INDEX")
    logging.info("=" * 70)

    if not CANDIDATE_NEOLOGISMS_FILE.exists():
        logging.error(f"Prerequisite not found: {CANDIDATE_NEOLOGISMS_FILE}")
        logging.error("Run stage 2 first.")
        return False

    logging.info(f"Loading candidate tokens from {CANDIDATE_NEOLOGISMS_FILE}...")
    target_tokens = load_candidate_tokens(CANDIDATE_NEOLOGISMS_FILE)
    logging.info(f"Loaded {len(target_tokens):,} tokens to search for")

    if len(target_tokens) == 0:
        logging.error("No candidate tokens to process. Pipeline cannot continue.")
        logging.error("Check stage 2 - all tokens were filtered out.")
        return False

    csv_files = get_all_csv_files()

    if TEST_MODE:
        csv_files = csv_files[:TEST_FILES_LIMIT]
        logging.info(f"TEST MODE: Processing only {len(csv_files)} files")
    else:
        logging.info(f"Found {len(csv_files)} CSV files")

    if not csv_files:
        logging.error("No CSV files found!")
        return False

    max_contexts = CONTEXT_CONFIG["max_contexts_per_token"]
    min_token_count = CONTEXT_CONFIG.get("min_token_count", 15)
    max_token_count = CONTEXT_CONFIG.get("max_token_count", 100)
    num_workers = CONTEXT_CONFIG["n_workers"] or min(os.cpu_count(), 16)

    logging.info(f"Using {num_workers} parallel workers")
    logging.info(f"Max contexts per token: {max_contexts}")
    logging.info(f"Min token count: {min_token_count}")
    logging.info(f"Max token count: {max_token_count}")

    dup_ids_arr = load_duplicate_ids_numpy()

    checkpoint_result = load_checkpoint(CHECKPOINTS_DIR) if not force else None
    if checkpoint_result:
        processed_files_set, tokens_found, partial_num = checkpoint_result
        csv_files_to_process = [f for f in csv_files if str(f) not in processed_files_set]
        processed_file_list = list(processed_files_set)
        logging.info(f"Remaining files to process: {len(csv_files_to_process)}")
    else:
        csv_files_to_process = csv_files
        tokens_found = set()
        processed_file_list = []
        partial_num = 0

    file_paths = [str(f) for f in csv_files_to_process]
    contexts_buffer = []
    files_processed_this_run = 0
    BATCH_SIZE = 500

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(target_tokens, dup_ids_arr, max_contexts, max_token_count),
    ) as executor:
        with tqdm(total=len(csv_files), desc="Processing files",
                  initial=len(processed_file_list)) as pbar:
            for batch_start in range(0, len(file_paths), BATCH_SIZE):
                batch = file_paths[batch_start:batch_start + BATCH_SIZE]
                futures = {executor.submit(process_single_file, fp): fp for fp in batch}

                for future in as_completed(futures):
                    filepath = futures[future]
                    try:
                        contexts = future.result()
                        if contexts:
                            contexts_buffer.append(contexts)
                            tokens_found.update(contexts.keys())
                        processed_file_list.append(filepath)
                    except Exception:
                        pass
                    pbar.update(1)
                    pbar.set_postfix({"tokens_found": len(tokens_found)})
                    files_processed_this_run += 1

                if contexts_buffer:
                    partial_num += 1
                    flush_contexts_to_disk(contexts_buffer, partial_num)
                    contexts_buffer = []
                save_checkpoint(processed_file_list, tokens_found, partial_num, CHECKPOINTS_DIR)

    logging.info("Loading partial files and merging contexts...")
    all_contexts = load_all_partials()
    merged_contexts = merge_contexts(all_contexts, max_contexts, min_token_count)

    logging.info(f"Saving context index to {CONTEXT_INDEX_FILE}...")
    with open(CONTEXT_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_contexts, f, indent=2, ensure_ascii=False)

    tokens_with_context = len(merged_contexts)
    tokens_without_context = len(target_tokens) - tokens_with_context
    total_contexts = sum(len(v) for v in merged_contexts.values())

    clean_checkpoint(CHECKPOINTS_DIR)
    mark_complete()

    logging.info("=" * 70)
    logging.info("STAGE 6 COMPLETE!")
    logging.info(f"Total target tokens:      {len(target_tokens):,}")
    logging.info(f"Tokens with context:      {tokens_with_context:,} ({100*tokens_with_context/len(target_tokens):.1f}%)")
    logging.info(f"Tokens without context:   {tokens_without_context:,}")
    logging.info(f"Total contexts collected: {total_contexts:,}")
    logging.info(f"Average contexts/token:   {total_contexts/max(1,tokens_with_context):.2f}")
    logging.info(f"Output: {CONTEXT_INDEX_FILE}")
    logging.info("=" * 70)

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 6: Build Context Index")
    parser.add_argument("--force", action="store_true", help="Force re-run even if complete")
    args = parser.parse_args()

    run(force=args.force)
