#!/usr/bin/env python3

import ctypes
import gc
import gzip
import logging
import platform
from pathlib import Path

from tqdm import tqdm

def get_memory_usage_gb():
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 3)
    except ImportError:
        return None

def aggressive_memory_cleanup():
    gc.collect()

    system = platform.system()
    try:
        if system == "Linux":
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            logging.info("  [Memory released to OS via malloc_trim]")
    except Exception:
        pass

def wait_for_memory_release(max_wait_seconds=60, check_interval=2):
    import time

    initial_mem = get_memory_usage_gb()
    if initial_mem is None:
        logging.info("  [psutil not available, skipping memory monitoring]")
        return

    logging.info(f"  [Memory before cleanup wait: {initial_mem:.2f} GB]")

    prev_mem = initial_mem
    stable_count = 0
    waited = 0

    while waited < max_wait_seconds:
        time.sleep(check_interval)
        waited += check_interval

        current_mem = get_memory_usage_gb()
        if current_mem is None:
            break

        if abs(current_mem - prev_mem) < (prev_mem * 0.01):
            stable_count += 1
            if stable_count >= 2:
                logging.info(f"  [Memory stabilized at {current_mem:.2f} GB after {waited}s]")
                break
        else:
            stable_count = 0

        logging.info(f"  [Memory: {current_mem:.2f} GB (waited {waited}s)]")
        prev_mem = current_mem

    final_mem = get_memory_usage_gb()
    if final_mem is not None:
        released = initial_mem - final_mem
        logging.info(f"  [Memory released: {released:.2f} GB ({initial_mem:.2f} -> {final_mem:.2f} GB)]")

from config import (
    CHECKPOINTS_DIR,
    LOG_DIR,
    UNIQUE_TOKENS_FILE,
    CANDIDATE_NEOLOGISMS_FILE,
    EXCLUSIONS_DIR,
    VOCAB_FILES,
    VOCAB_FILTERING_CONFIG,
)

from utils.filtering_utils import is_valid_candidate, load_stopwords

def setup_logging():
    log_file = LOG_DIR / "stage_4_vocab_filtering.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_vocabularies(vocab_files):
    combined_vocab = set()

    logging.info("Loading vocabulary files...")
    for vocab_file in vocab_files:
        vocab_path = Path(vocab_file)
        if not vocab_path.exists():
            logging.warning(f"Vocabulary file not found: {vocab_file}")
            continue

        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                words = {line.strip().lower() for line in f if line.strip()}
                combined_vocab.update(words)
                logging.info(f"  Loaded {len(words):,} words from {vocab_path.name}")
        except Exception as e:
            logging.error(f"Error loading {vocab_file}: {e}")

    logging.info(f"Total combined vocabulary: {len(combined_vocab):,} words")
    return combined_vocab

def load_unique_tokens():
    logging.info(f"Loading unique tokens from {UNIQUE_TOKENS_FILE}...")

    tokens = set()
    with gzip.open(UNIQUE_TOKENS_FILE, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading tokens"):
            token = line.strip()
            if token:
                tokens.add(token)

    logging.info(f"Loaded {len(tokens):,} unique tokens")
    return tokens

def step1_vocab_filtering(tokens, vocab):
    logging.info("Step 1: Vocabulary filtering...")

    candidates = set()
    filtered_tokens = []

    for token in tqdm(tokens, desc="Filtering against vocabulary"):
        if token.lower() not in vocab:
            candidates.add(token.lower())
        else:
            filtered_tokens.append(token.lower())

    logging.info(f"  Tokens in vocabulary (filtered out): {len(filtered_tokens):,}")
    logging.info(f"  Candidate tokens remaining: {len(candidates):,}")

    with open(CANDIDATE_NEOLOGISMS_FILE, 'w', encoding='utf-8') as f:
        for token in sorted(candidates):
            f.write(f"{token}\n")

    vocab_filtered_log = EXCLUSIONS_DIR / "vocab_filtered.txt"
    with open(vocab_filtered_log, 'w', encoding='utf-8') as f:
        f.write("# Tokens filtered out because they exist in vocabulary\n")
        for token in sorted(filtered_tokens):
            f.write(f"{token}\n")

    logging.info(f"  Saved candidates to {CANDIDATE_NEOLOGISMS_FILE}")
    logging.info(f"  Exclusion log: {vocab_filtered_log}")
    return candidates

def step2_pattern_cleaning(candidates):
    logging.info("Step 2: Pattern cleaning...")

    stopwords = load_stopwords()
    cleaned = []
    filtered_tokens = []

    for token in tqdm(candidates, desc="Cleaning patterns"):
        if is_valid_candidate(token, stopwords):
            cleaned.append(token)
        else:
            filtered_tokens.append(token)

    logging.info(f"  Removed by pattern rules: {len(filtered_tokens):,}")
    logging.info(f"  Cleaned candidates: {len(cleaned):,}")

    with open(CANDIDATE_NEOLOGISMS_FILE, 'w', encoding='utf-8') as f:
        for token in sorted(cleaned):
            f.write(f"{token}\n")

    pattern_filtered_log = EXCLUSIONS_DIR / "pattern_filtered.txt"
    with open(pattern_filtered_log, 'w', encoding='utf-8') as f:
        f.write("# Tokens filtered out by pattern rules\n")
        for token in sorted(filtered_tokens):
            f.write(f"{token}\n")

    logging.info(f"  Updated {CANDIDATE_NEOLOGISMS_FILE}")
    logging.info(f"  Exclusion log: {pattern_filtered_log}")
    return cleaned


_worker_sym_spell = None
_worker_vocab_lower = None
_worker_min_word_length = None

def _init_segmentation_worker(freq_dict_path, vocab_lower_set, min_word_length):
    global _worker_sym_spell, _worker_vocab_lower, _worker_min_word_length
    from symspellpy import SymSpell

    _worker_vocab_lower = vocab_lower_set
    _worker_min_word_length = min_word_length

    max_edit_distance = VOCAB_FILTERING_CONFIG["max_edit_distance"]
    _worker_sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
    _worker_sym_spell.load_dictionary(str(freq_dict_path), term_index=0, count_index=1, separator="\t")

def _segment_token(token):
    global _worker_sym_spell, _worker_vocab_lower, _worker_min_word_length

    if len(token) < _worker_min_word_length:
        return (token, None)

    result = _worker_sym_spell.word_segmentation(token)
    segmented = result.corrected_string

    if ' ' in segmented:
        parts = segmented.split()

        all_parts_in_vocab = all(part.lower() in _worker_vocab_lower for part in parts)

        if all_parts_in_vocab:
            return (token, segmented)

    return (token, None)

def step3_word_segmentation(candidates, vocab):
    import multiprocessing as mp

    logging.info("Step 3: Word segmentation (merged words detection)...")

    try:
        from symspellpy import SymSpell
    except ImportError:
        logging.warning("symspellpy not installed. Skipping word segmentation.")
        return candidates

    vocab_lower = frozenset(w.lower() for w in vocab)

    min_word_length = VOCAB_FILTERING_CONFIG.get("min_word_length_segmentation", 6)

    from config import VOCAB_DIR
    freq_dict_path = VOCAB_DIR / "symspell_frequency_dict.txt"

    if not freq_dict_path.exists():
        logging.warning(f"  Frequency dictionary not found: {freq_dict_path}")
        logging.warning("  Skipping word segmentation.")
        return candidates

    step3_progress_file = CHECKPOINTS_DIR / "stage4_step3_progress.jsonl"
    step3_kept_file = CHECKPOINTS_DIR / "stage4_step3_kept.txt"
    step3_merged_file = CHECKPOINTS_DIR / "stage4_step3_merged.tsv"

    already_processed = set()
    kept = []
    merged_words = []

    if step3_kept_file.exists():
        logging.info(f"  Resuming from checkpoint: {step3_kept_file}")
        with open(step3_kept_file, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token:
                    kept.append(token)
                    already_processed.add(token)
        logging.info(f"  Loaded {len(kept):,} kept tokens from checkpoint")

    if step3_merged_file.exists():
        with open(step3_merged_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2:
                    merged_words.append((parts[0], parts[1]))
                    already_processed.add(parts[0])
        logging.info(f"  Loaded {len(merged_words):,} merged words from checkpoint")

    remaining = [t for t in candidates if t not in already_processed]
    logging.info(f"  Already processed: {len(already_processed):,}, remaining: {len(remaining):,}")

    n_workers = min(mp.cpu_count(), 16)
    logging.info(f"  Using {n_workers} worker processes for segmentation...")

    CHECKPOINT_INTERVAL = 500_000
    since_checkpoint = 0

    f_kept = open(step3_kept_file, 'a', encoding='utf-8')
    f_merged = open(step3_merged_file, 'a', encoding='utf-8')

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_init_segmentation_worker,
            initargs=(freq_dict_path, vocab_lower, min_word_length)
        ) as pool:
            results = pool.imap_unordered(_segment_token, remaining, chunksize=1000)

            for token, segmented in tqdm(results, total=len(remaining), desc="Detecting merged words"):
                if segmented is None:
                    kept.append(token)
                    f_kept.write(token + '\n')
                else:
                    merged_words.append((token, segmented))
                    f_merged.write(f"{token}\t{segmented}\n")

                since_checkpoint += 1
                if since_checkpoint >= CHECKPOINT_INTERVAL:
                    f_kept.flush()
                    f_merged.flush()
                    logging.info(f"  Checkpoint: {len(kept):,} kept, {len(merged_words):,} merged")
                    since_checkpoint = 0
    finally:
        f_kept.close()
        f_merged.close()

    logging.info(f"  Merged words detected: {len(merged_words):,}")
    logging.info(f"  Candidates remaining: {len(kept):,}")

    with open(CANDIDATE_NEOLOGISMS_FILE, 'w', encoding='utf-8') as f:
        for token in sorted(kept):
            f.write(f"{token}\n")

    merged_log = EXCLUSIONS_DIR / "merged_words.txt"
    with open(merged_log, 'w', encoding='utf-8') as f:
        f.write("# Merged words detected via segmentation: original -> segmented\n")
        for token, segmented in sorted(merged_words):
            f.write(f"{token}\t{segmented}\n")

    logging.info(f"  Updated {CANDIDATE_NEOLOGISMS_FILE}")
    logging.info(f"  Exclusion log: {merged_log}")

    step3_kept_file.unlink(missing_ok=True)
    step3_merged_file.unlink(missing_ok=True)

    return kept

_typo_worker_sym_spell = None
_typo_worker_max_edit_distance = None
_typo_worker_min_word_length = None

def _init_typo_worker(freq_dict_path, dataset_dict_path, max_edit_distance, min_word_length):
    global _typo_worker_sym_spell, _typo_worker_max_edit_distance, _typo_worker_min_word_length
    from symspellpy import SymSpell
    from pathlib import Path

    _typo_worker_max_edit_distance = max_edit_distance
    _typo_worker_min_word_length = min_word_length

    _typo_worker_sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
    _typo_worker_sym_spell.load_dictionary(str(freq_dict_path), term_index=0, count_index=1, separator="\t")
    if dataset_dict_path and Path(dataset_dict_path).exists():
        _typo_worker_sym_spell.load_dictionary(str(dataset_dict_path), term_index=0, count_index=1, separator="\t")

def _check_typo(token):
    global _typo_worker_sym_spell, _typo_worker_max_edit_distance, _typo_worker_min_word_length
    from symspellpy import Verbosity

    if len(token) < _typo_worker_min_word_length:
        return (token, None, None)

    suggestions = _typo_worker_sym_spell.lookup(
        token,
        Verbosity.CLOSEST,
        max_edit_distance=_typo_worker_max_edit_distance
    )

    if suggestions:
        best = suggestions[0]
        if (best.term != token.lower() and
            best.distance > 0 and
            best.distance <= _typo_worker_max_edit_distance and
            best.distance / len(token) < 0.3 and
            best.count > 100):
            return (token, best.term, best.distance)

    return (token, None, None)

def step4_typo_filtering(candidates):
    import multiprocessing as mp

    logging.info("Step 4: Typo filtering...")

    try:
        from symspellpy import SymSpell, Verbosity
    except ImportError:
        logging.warning("symspellpy not installed. Skipping typo filtering.")
        return candidates

    max_edit_distance = VOCAB_FILTERING_CONFIG["max_edit_distance"]
    min_word_length = VOCAB_FILTERING_CONFIG["min_word_length_typo"]

    from config import VOCAB_DIR
    freq_dict_path = VOCAB_DIR / "symspell_frequency_dict.txt"
    dataset_dict_path = VOCAB_DIR / "dataset_frequency_dict_deduped.txt"

    if not freq_dict_path.exists():
        logging.warning(f"  Frequency dictionary not found: {freq_dict_path}")
        logging.warning("  Skipping typo filtering.")
        return candidates

    if dataset_dict_path.exists():
        logging.info(f"  Dataset frequency dict: {dataset_dict_path}")
    else:
        logging.info("  No dataset frequency dict found, using standard dict only")
        dataset_dict_path = None

    n_workers = min(mp.cpu_count(), 8)
    logging.info(f"  Using {n_workers} worker processes for typo detection...")

    step4_kept_file = CHECKPOINTS_DIR / "stage4_step4_kept.txt"
    step4_typos_file = CHECKPOINTS_DIR / "stage4_step4_typos.tsv"

    already_processed = set()
    kept = []
    typos_found = []

    if step4_kept_file.exists():
        logging.info(f"  Resuming from checkpoint: {step4_kept_file}")
        with open(step4_kept_file, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token:
                    kept.append(token)
                    already_processed.add(token)
        logging.info(f"  Loaded {len(kept):,} kept tokens from checkpoint")

    if step4_typos_file.exists():
        with open(step4_typos_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 3:
                    typos_found.append((parts[0], parts[1], int(parts[2])))
                    already_processed.add(parts[0])
        logging.info(f"  Loaded {len(typos_found):,} typos from checkpoint")

    remaining = [t for t in candidates if t not in already_processed]
    logging.info(f"  Already processed: {len(already_processed):,}, remaining: {len(remaining):,}")

    CHECKPOINT_INTERVAL = 500_000
    since_checkpoint = 0

    f_kept = open(step4_kept_file, 'a', encoding='utf-8')
    f_typos = open(step4_typos_file, 'a', encoding='utf-8')

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_init_typo_worker,
            initargs=(freq_dict_path, dataset_dict_path, max_edit_distance, min_word_length)
        ) as pool:
            results = pool.imap_unordered(_check_typo, remaining, chunksize=1000)

            for token, correction, distance in tqdm(results, total=len(remaining), desc="Detecting typos"):
                if correction is None:
                    kept.append(token)
                    f_kept.write(token + '\n')
                else:
                    typos_found.append((token, correction, distance))
                    f_typos.write(f"{token}\t{correction}\t{distance}\n")

                since_checkpoint += 1
                if since_checkpoint >= CHECKPOINT_INTERVAL:
                    f_kept.flush()
                    f_typos.flush()
                    logging.info(f"  Checkpoint: {len(kept):,} kept, {len(typos_found):,} typos")
                    since_checkpoint = 0
    finally:
        f_kept.close()
        f_typos.close()

    logging.info(f"  Typos detected: {len(typos_found):,}")
    logging.info(f"  Final candidates: {len(kept):,}")

    with open(CANDIDATE_NEOLOGISMS_FILE, 'w', encoding='utf-8') as f:
        for token in sorted(kept):
            f.write(f"{token}\n")

    typos_log = EXCLUSIONS_DIR / "typos_detected.txt"
    with open(typos_log, 'w', encoding='utf-8') as f:
        f.write("# Detected typos: original → correction (edit_distance)\n")
        for token, correction, dist in sorted(typos_found):
            f.write(f"{token}\t{correction}\t{dist}\n")

    logging.info(f"  Updated {CANDIDATE_NEOLOGISMS_FILE}")
    logging.info(f"  Exclusion log: {typos_log}")

    step4_kept_file.unlink(missing_ok=True)
    step4_typos_file.unlink(missing_ok=True)

    return kept

STEP_FLAGS = {
    1: CHECKPOINTS_DIR / "stage4_step1_vocab.flag",
    2: CHECKPOINTS_DIR / "stage4_step2_pattern.flag",
    3: CHECKPOINTS_DIR / "stage4_step3_segmentation.flag",
    4: CHECKPOINTS_DIR / "stage4_step4_typo.flag",
}


def is_complete():
    flag_file = CHECKPOINTS_DIR / "stage4_vocab_complete.flag"
    return flag_file.exists()


def mark_complete():
    checkpoint_file = CHECKPOINTS_DIR / "stage4_vocab_complete.flag"
    checkpoint_file.touch()


def load_candidates_from_file():
    candidates = []
    with open(CANDIDATE_NEOLOGISMS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                candidates.append(token)
    return candidates

def run(force=False):
    setup_logging()

    if is_complete() and not force:
        logging.info("Stage 4 already complete. Skipping. Use force=True to re-run.")
        return True

    logging.info("=" * 70)
    logging.info("STAGE 4: VOCABULARY FILTERING")
    logging.info("=" * 70)

    if not UNIQUE_TOKENS_FILE.exists():
        logging.error(f"Prerequisite not found: {UNIQUE_TOKENS_FILE}")
        logging.error("Run stage 1 first.")
        return False

    input_count = None
    after_vocab_count = None
    after_pattern_count = None
    after_segmentation_count = None

    vocab_needed_for_step3 = not STEP_FLAGS[3].exists()

    if not STEP_FLAGS[1].exists():
        tokens = load_unique_tokens()
        vocab = load_vocabularies(VOCAB_FILES)
        input_count = len(tokens)

        candidates = step1_vocab_filtering(tokens, vocab)
        after_vocab_count = len(candidates)

        STEP_FLAGS[1].touch()
        del tokens
        gc.collect()
        logging.info("  [Memory cleanup: tokens freed]")
    else:
        logging.info("Step 1 already complete, loading candidates from file...")
        candidates = load_candidates_from_file()
        after_vocab_count = len(candidates)
        vocab = load_vocabularies(VOCAB_FILES) if vocab_needed_for_step3 else None

    if not STEP_FLAGS[2].exists():
        cleaned = step2_pattern_cleaning(candidates)
        after_pattern_count = len(cleaned)
        STEP_FLAGS[2].touch()

        del candidates
        gc.collect()
        logging.info("  [Memory cleanup: candidates freed]")
    else:
        logging.info("Step 2 already complete, loading candidates from file...")
        cleaned = load_candidates_from_file()
        after_pattern_count = len(cleaned)
        del candidates
        gc.collect()

    if not STEP_FLAGS[3].exists():
        segmented = step3_word_segmentation(cleaned, vocab)
        after_segmentation_count = len(segmented)
        STEP_FLAGS[3].touch()

        del cleaned
        gc.collect()
        logging.info("  [Memory cleanup: cleaned freed]")

        if vocab is not None:
            del vocab
            gc.collect()
            logging.info("  [Memory cleanup: vocab freed]")
    else:
        logging.info("Step 3 already complete, loading candidates from file...")
        segmented = load_candidates_from_file()
        after_segmentation_count = len(segmented)
        del cleaned
        gc.collect()
        if vocab is not None:
            del vocab
            gc.collect()

    logging.info("  [Aggressive memory cleanup before typo filtering...]")
    aggressive_memory_cleanup()
    wait_for_memory_release(max_wait_seconds=60, check_interval=2)

    if not STEP_FLAGS[4].exists():
        final = step4_typo_filtering(segmented)
        STEP_FLAGS[4].touch()
    else:
        logging.info("Step 4 already complete, loading candidates from file...")
        final = load_candidates_from_file()

    del segmented
    gc.collect()
    logging.info("  [Memory cleanup: segmented freed]")

    mark_complete()

    logging.info("Cleaning up step checkpoints...")
    for flag in STEP_FLAGS.values():
        if flag.exists():
            flag.unlink()

    logging.info("=" * 70)
    logging.info("STAGE 4 COMPLETE!")
    logging.info(f"Input tokens: {input_count:,}")
    logging.info(f"After vocab filter: {after_vocab_count:,}")
    logging.info(f"After pattern cleaning: {after_pattern_count:,}")
    logging.info(f"After word segmentation: {after_segmentation_count:,}")
    logging.info(f"Final candidates (after typo filter): {len(final):,}")
    logging.info("=" * 70)

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Vocabulary Filtering")
    parser.add_argument("--force", action="store_true", help="Force re-run even if complete")
    args = parser.parse_args()

    run(force=args.force)
