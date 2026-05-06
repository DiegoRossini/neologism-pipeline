#!/usr/bin/env python3

import json
import logging
import argparse
import tempfile
from pathlib import Path
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR,
    EXCLUSIONS_DIR,
    CANDIDATE_NEOLOGISMS_FILE,
    CHECKPOINTS_DIR,
    LOG_DIR,
    LOG_FORMAT,
    TOKEN_COUNTS_FILE,
)
from language_constants import RELEVANT_LANGUAGES

FREQUENCY_FILTERED_FILE = OUTPUT_DIR / "frequency_filtered_candidates.txt"
WAITING_LIST_FILE = OUTPUT_DIR / "waiting_list_candidates.txt"
REINTEGRATED_FILE = OUTPUT_DIR / "reintegrated_candidates.txt"
FOREIGN_WORDS_FILE = OUTPUT_DIR / "foreign_words_detected.txt"

MIN_OCCURRENCES = 100

LANG_CONFIDENCE_THRESHOLD = 0.75

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "stage_5_frequency_filtering.log"

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_token_counts():
    deduped_file = OUTPUT_DIR / "token_frequencies.tsv"
    if deduped_file.exists():
        logging.info(f"Loading deduped token counts from {deduped_file}...")
        token_totals = {}
        with open(deduped_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in tqdm(f, desc="Loading token counts"):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    token_totals[parts[0]] = int(parts[2])
        logging.info(f"  Loaded {len(token_totals):,} tokens with deduped counts")
        return token_totals

    logging.info(f"Loading raw token counts from {TOKEN_COUNTS_FILE}...")
    if not TOKEN_COUNTS_FILE.exists():
        logging.error(f"Token counts file not found: {TOKEN_COUNTS_FILE}")
        return {}

    token_totals = {}
    logging.info("  Counting lines...")
    with open(TOKEN_COUNTS_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    logging.info(f"  Total tokens in counts file: {total_lines:,}")

    with open(TOKEN_COUNTS_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Loading token counts"):
            try:
                entry = json.loads(line.strip())
                token = entry['token']
                counts = entry['counts']
                total = sum(counts.values())
                token_totals[token] = total
            except (json.JSONDecodeError, KeyError):
                continue

    logging.info(f"  Loaded {len(token_totals):,} tokens with raw counts (no dedup file found)")
    return token_totals

def load_candidates():
    logging.info(f"Loading candidates from {CANDIDATE_NEOLOGISMS_FILE}...")

    if not CANDIDATE_NEOLOGISMS_FILE.exists():
        logging.error(f"Candidates file not found: {CANDIDATE_NEOLOGISMS_FILE}")
        return set()

    with open(CANDIDATE_NEOLOGISMS_FILE, 'r', encoding='utf-8') as f:
        candidates = set(line.strip() for line in f if line.strip())

    logging.info(f"  Loaded {len(candidates):,} candidates")
    return candidates

def load_excluded_words():
    excluded = {}

    typos_file = EXCLUSIONS_DIR / "typos_detected.txt"
    if typos_file.exists():
        logging.info(f"  Loading excluded typos from {typos_file.name}...")
        with open(typos_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    excluded[token] = ('typo', parts[1] if len(parts) > 1 else '')
        logging.info(f"    Loaded {len([k for k, v in excluded.items() if v[0] == 'typo']):,} typos")

    merged_file = EXCLUSIONS_DIR / "merged_words.txt"
    if merged_file.exists():
        logging.info(f"  Loading excluded merged words from {merged_file.name}...")
        count_before = len(excluded)
        with open(merged_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    if token not in excluded:
                        excluded[token] = ('merged', parts[1] if len(parts) > 1 else '')
        logging.info(f"    Loaded {len(excluded) - count_before:,} merged words")

    logging.info(f"  Total excluded words: {len(excluded):,}")
    return excluded

REINTEGRATION_TYPO_MIN_FREQ = 500
REINTEGRATION_TYPO_MAX_EDIT_DISTANCE = 2
REINTEGRATION_TYPO_MIN_WORD_LENGTH = 5

_reint_typo_sym_spell = None

def _init_reint_typo_worker(dict_path):
    global _reint_typo_sym_spell
    from symspellpy import SymSpell
    _reint_typo_sym_spell = SymSpell(max_dictionary_edit_distance=REINTEGRATION_TYPO_MAX_EDIT_DISTANCE, prefix_length=7)
    _reint_typo_sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1, separator="\t")

def _check_reint_typo(token):
    global _reint_typo_sym_spell
    from symspellpy import Verbosity

    if len(token) < REINTEGRATION_TYPO_MIN_WORD_LENGTH:
        return (token, None, None)

    suggestions = _reint_typo_sym_spell.lookup(
        token, Verbosity.CLOSEST,
        max_edit_distance=REINTEGRATION_TYPO_MAX_EDIT_DISTANCE
    )
    if suggestions:
        best = suggestions[0]
        if (best.term != token.lower() and
            best.distance > 0 and
            best.distance / len(token) < 0.3 and
            best.count > 100):
            return (token, best.term, best.distance)
    return (token, None, None)

def dedup_reintegrated_typos(reintegrated, token_counts):
    import multiprocessing as mp

    if not reintegrated:
        return reintegrated, []

    reference_tokens = [(t, c) for t, c, r, d in reintegrated if c >= REINTEGRATION_TYPO_MIN_FREQ]
    to_check = [(t, c, r, d) for t, c, r, d in reintegrated if c < REINTEGRATION_TYPO_MIN_FREQ]

    if not reference_tokens or not to_check:
        logging.info(f"  Skipping reintegration dedup: {len(reference_tokens)} reference, {len(to_check)} to check")
        return reintegrated, []

    logging.info(f"  Reference tokens (freq >= {REINTEGRATION_TYPO_MIN_FREQ}): {len(reference_tokens)}")
    logging.info(f"  Tokens to check: {len(to_check)}")

    tmp_dict = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=str(OUTPUT_DIR))
    try:
        for token, count in reference_tokens:
            tmp_dict.write(f"{token}\t{count}\n")
        tmp_dict.close()

        n_workers = min(mp.cpu_count(), 16)
        tokens_to_check = [t for t, c, r, d in to_check]

        typos_found = []

        with mp.Pool(
            processes=n_workers,
            initializer=_init_reint_typo_worker,
            initargs=(tmp_dict.name,)
        ) as pool:
            for token, correction, distance in tqdm(
                pool.imap_unordered(_check_reint_typo, tokens_to_check, chunksize=500),
                total=len(tokens_to_check),
                desc="Dedup reintegrated typos"
            ):
                if correction is not None:
                    typos_found.append((token, correction, distance))

        typo_tokens = set(t for t, c, d in typos_found)
        surviving = [(t, c, r, d) for t, c, r, d in reintegrated if t not in typo_tokens]

        logging.info(f"  Reintegrated typos caught: {len(typos_found)}")
        if typos_found:
            examples = sorted(typos_found, key=lambda x: token_counts.get(x[1], 0), reverse=True)[:10]
            for token, correction, dist in examples:
                logging.info(f"    {token} -> {correction} (dist={dist}, ref_freq={token_counts.get(correction, 0):,})")

        reint_typo_log = EXCLUSIONS_DIR / "reintegrated_typos_caught.txt"
        with open(reint_typo_log, 'w', encoding='utf-8') as f:
            for token, correction, dist in sorted(typos_found, key=lambda x: x[1]):
                f.write(f"{token}\t{correction}\t{dist}\n")
        logging.info(f"  Report: {reint_typo_log}")

        return surviving, typos_found
    finally:
        Path(tmp_dict.name).unlink(missing_ok=True)

def detect_foreign_words(candidates):
    logging.info("Initializing Lingua language detector...")
    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()

    english_candidates = []
    foreign_words = []

    logging.info(f"Detecting foreign words among {len(candidates):,} candidates...")

    for token in tqdm(candidates, desc="Language detection"):
        lang = detector.detect_language_of(token)

        if lang is None:
            english_candidates.append(token)
            continue

        if lang == Language.ENGLISH:
            english_candidates.append(token)
            continue

        if lang not in RELEVANT_LANGUAGES:
            english_candidates.append(token)
            continue

        conf = detector.compute_language_confidence(token, lang)

        if conf >= LANG_CONFIDENCE_THRESHOLD:
            foreign_words.append((token, lang.name, conf))
        else:
            english_candidates.append(token)

    logging.info(f"  English/uncertain: {len(english_candidates):,}")
    logging.info(f"  Foreign detected: {len(foreign_words):,}")

    return set(english_candidates), foreign_words

def run_frequency_filtering(force=False):
    logging.info("=" * 70)
    logging.info("STAGE 5: FREQUENCY FILTERING")
    logging.info("=" * 70)

    if is_complete() and not force:
        logging.info("Stage 5 already complete. Use force=True to re-run.")
        return True

    token_counts = load_token_counts()
    if not token_counts:
        logging.error("Failed to load token counts. Aborting.")
        return False

    candidates = load_candidates()
    if not candidates:
        logging.error("No candidates to filter. Aborting.")
        return False

    excluded = load_excluded_words()

    logging.info(f"\nStep 1: Filtering candidates with <{MIN_OCCURRENCES} occurrences...")

    high_freq_candidates = []
    low_freq_candidates = []
    no_count_candidates = []

    for token in tqdm(candidates, desc="Filtering by frequency"):
        count = token_counts.get(token, 0)
        if count >= MIN_OCCURRENCES:
            high_freq_candidates.append((token, count))
        elif count > 0:
            low_freq_candidates.append((token, count))
        else:
            no_count_candidates.append(token)

    logging.info(f"  High frequency (>={MIN_OCCURRENCES}): {len(high_freq_candidates):,}")
    logging.info(f"  Low frequency (1-{MIN_OCCURRENCES-1}): {len(low_freq_candidates):,}")
    logging.info(f"  No count data: {len(no_count_candidates):,}")

    logging.info(f"\nStep 2: Re-integrating excluded words with >={MIN_OCCURRENCES} occurrences...")

    reintegrated = []
    for token, (reason, detail) in tqdm(excluded.items(), desc="Checking excluded words"):
        count = token_counts.get(token, 0)
        if count >= MIN_OCCURRENCES:
            reintegrated.append((token, count, reason, detail))

    logging.info(f"  Reintegrated: {len(reintegrated):,}")

    logging.info(f"\nStep 2b: Second-pass typo dedup on reintegrated tokens...")
    reintegrated, reint_typos = dedup_reintegrated_typos(reintegrated, token_counts)
    logging.info(f"  Reintegrated after dedup: {len(reintegrated):,}")

    final_candidates = set(t for t, c in high_freq_candidates)
    final_candidates.update(t for t, c, r, d in reintegrated)

    logging.info(f"\nStep 3: Detecting and filtering foreign language words...")
    final_candidates, foreign_words = detect_foreign_words(final_candidates)

    logging.info("\nStep 4: Saving results...")

    with open(FREQUENCY_FILTERED_FILE, 'w', encoding='utf-8') as f:
        for token in sorted(final_candidates):
            f.write(f"{token}\n")
    logging.info(f"  Saved {len(final_candidates):,} candidates to {FREQUENCY_FILTERED_FILE.name}")

    with open(WAITING_LIST_FILE, 'w', encoding='utf-8') as f:
        f.write("# Waiting list: candidates with <100 occurrences\n")
        f.write("# Format: token\\tcount\n")
        for token, count in sorted(low_freq_candidates, key=lambda x: -x[1]):
            f.write(f"{token}\t{count}\n")
        f.write("# No count data:\n")
        for token in sorted(no_count_candidates):
            f.write(f"{token}\t0\n")
    logging.info(f"  Saved {len(low_freq_candidates) + len(no_count_candidates):,} to waiting list")

    with open(REINTEGRATED_FILE, 'w', encoding='utf-8') as f:
        f.write("# Reintegrated words: previously excluded but have high frequency\n")
        f.write("# Format: token\\tcount\\treason\\tdetail\n")
        for token, count, reason, detail in sorted(reintegrated, key=lambda x: -x[1]):
            f.write(f"{token}\t{count}\t{reason}\t{detail}\n")
    logging.info(f"  Saved {len(reintegrated):,} reintegrated words log")

    with open(FOREIGN_WORDS_FILE, 'w', encoding='utf-8') as f:
        f.write("# Foreign words detected by Lingua language detector\n")
        f.write(f"# Confidence threshold: {LANG_CONFIDENCE_THRESHOLD}\n")
        f.write("# Format: token\\tlanguage\\tconfidence\n")
        for token, lang, conf in sorted(foreign_words, key=lambda x: (-x[2], x[1], x[0])):
            f.write(f"{token}\t{lang}\t{conf:.3f}\n")
    logging.info(f"  Saved {len(foreign_words):,} foreign words to {FOREIGN_WORDS_FILE.name}")

    with open(CANDIDATE_NEOLOGISMS_FILE, 'w', encoding='utf-8') as f:
        for token in sorted(final_candidates):
            f.write(f"{token}\n")
    logging.info(f"  Updated {CANDIDATE_NEOLOGISMS_FILE.name}")

    logging.info("\n" + "=" * 70)
    logging.info("STAGE 5 COMPLETE!")
    logging.info(f"Input candidates: {len(candidates):,}")
    logging.info(f"High frequency candidates: {len(high_freq_candidates):,}")
    logging.info(f"Reintegrated from exclusions: {len(reintegrated):,}")
    logging.info(f"Foreign words filtered: {len(foreign_words):,}")
    logging.info(f"Final candidates: {len(final_candidates):,}")
    logging.info(f"Moved to waiting list: {len(low_freq_candidates) + len(no_count_candidates):,}")
    logging.info("=" * 70)

    mark_complete()
    return True

def run():
    setup_logging()
    return run_frequency_filtering()

def is_complete():
    return FREQUENCY_FILTERED_FILE.exists()

def mark_complete():
    checkpoint_file = CHECKPOINTS_DIR / "stage5_freq_complete.flag"
    checkpoint_file.touch()

def main():
    global MIN_OCCURRENCES

    parser = argparse.ArgumentParser(description="Stage 5: Frequency Filtering")
    parser.add_argument('--min-occurrences', type=int, default=100,
                        help="Minimum occurrences threshold (default: 100)")
    parser.add_argument('--force', action='store_true', help="Force re-run")
    args = parser.parse_args()

    MIN_OCCURRENCES = args.min_occurrences

    setup_logging()
    success = run_frequency_filtering(force=args.force)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
