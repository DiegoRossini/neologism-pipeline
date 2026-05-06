#!/usr/bin/env python3

import os
import sys
import time
import logging
import json
from pathlib import Path

BASE_COMMENTS = os.environ.get(
    "NEOLOGISM_TEST_CORPUS_DIR",
    str(Path(__file__).resolve().parent / "test_corpus"),
)
TEST_FILE_LIMIT = int(os.environ.get("NEOLOGISM_TEST_FILE_LIMIT", "20"))
TEST_FILES = sorted(str(p) for p in Path(BASE_COMMENTS).glob("*.csv.gz"))[:TEST_FILE_LIMIT]
if not TEST_FILES:
    raise RuntimeError(
        f"No *.csv.gz files found in {BASE_COMMENTS}. "
        f"Set NEOLOGISM_TEST_CORPUS_DIR to a directory containing test CSV.gz files, "
        f"or place files in {Path(__file__).resolve().parent / 'test_corpus'}/."
    )

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

import config

TEST_DIR = os.path.join(PIPELINE_DIR, "test_run")
os.makedirs(TEST_DIR, exist_ok=True)

test_data_dir = config.Path(TEST_DIR) / "test_data"

config.BASE_DIR = config.Path(TEST_DIR)
config.DATA_DIRS = [test_data_dir]
config.PIPELINE_DATA_DIR = config.Path(TEST_DIR) / "data"
config.TEMP_BATCHES_DIR = config.PIPELINE_DATA_DIR / "temp_batches"
config.CHECKPOINTS_DIR = config.PIPELINE_DATA_DIR / "checkpoints"
config.OUTPUT_DIR = config.PIPELINE_DATA_DIR / "output"
config.TOKEN_COUNTS_FILE = config.Path(TEST_DIR) / "token_counts_final.jsonl"
config.UNIQUE_TOKENS_FILE = config.OUTPUT_DIR / "unique_tokens.txt.gz"
config.CANDIDATE_NEOLOGISMS_FILE = config.OUTPUT_DIR / "candidate_neologisms.txt"
config.EXCLUSIONS_DIR = config.OUTPUT_DIR / "exclusions"
config.CONTEXT_INDEX_FILE = config.OUTPUT_DIR / "context_index.json"
config.CLASSIFICATION_DIR = config.OUTPUT_DIR / "classification_results"
config.FILTERED_NEOLOGISMS_FILE = config.OUTPUT_DIR / "filtered_neologisms.txt"
config.TOKEN_FREQUENCIES_FILE = config.OUTPUT_DIR / "token_frequencies.tsv"
config.UNIQUE_TOKENS_PER_DAY_FILE = config.OUTPUT_DIR / "unique_tokens_per_day.tsv"
config.UNIQUE_TOKENS_PER_SUBREDDIT_FILE = config.OUTPUT_DIR / "unique_tokens_per_subreddit.tsv"
config.TOKEN_COUNTING_STATS_FILE = config.OUTPUT_DIR / "token_counting_stats.json"
config.DUPLICATE_IDS_FILE = config.OUTPUT_DIR / "duplicate_ids.txt"
config.LOG_DIR = config.OUTPUT_DIR / "logs"
config.PIPELINE_STATE_FILE = config.OUTPUT_DIR / "pipeline_state.json"

config.VOCAB_DIR = config.Path(PIPELINE_DIR) / "vocabs"
config.VOCAB_FILES = [
    config.VOCAB_DIR / "wikipedia_titles_pre2015_vocab.txt",
    config.VOCAB_DIR / "wiktionary_pre2015_vocab.txt",
    config.VOCAB_DIR / "noslang_word_list.txt",
    config.VOCAB_DIR / "pre2015_vocab.txt",
    config.VOCAB_DIR / "urban_dict_pre2015_vocab.txt",
    config.VOCAB_DIR / "wordnet_vocab.txt",
]

config.ensure_directories()
test_data_dir.mkdir(parents=True, exist_ok=True)

for f in TEST_FILES:
    link = test_data_dir / os.path.basename(f)
    if not link.exists():
        os.symlink(f, link)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / "test_pipeline.log"),
        logging.StreamHandler()
    ]
)

logging.info("=" * 70)
logging.info(f"GITHUB PIPELINE TEST — {len(TEST_FILES)} files, stages 0→6")
logging.info(f"Test directory: {TEST_DIR}")
logging.info("=" * 70)

start = time.time()
import importlib

logging.info("\n>>> STAGE 0: Tokenization")
import stage_0_tokenization as s0
importlib.reload(s0)
s0.run(force=True)

logging.info("\n>>> STAGE 1: Merge batches")
import stage_1_merge_batches as s1
importlib.reload(s1)
s1.run(force=True)

logging.info("\n>>> STAGE 2: Dedup — SKIPPED (not needed for test)")

logging.info("\n>>> STAGE 3: Token counting")
import stage_3_token_counting as s3
importlib.reload(s3)
s3.run(force=True)

logging.info("\n>>> STAGE 4: Vocab filtering")
import stage_4_vocab_filtering as s4
importlib.reload(s4)
s4.run(force=True)

logging.info("\n>>> STAGE 5: Frequency filtering")
import stage_5_frequency_filtering as s5
importlib.reload(s5)
s5.MIN_OCCURRENCES = 100
s5.run()

logging.info("\n>>> STAGE 6: Build context")
import stage_6_build_context as s6
importlib.reload(s6)
s6.run(force=True)

if config.CANDIDATE_NEOLOGISMS_FILE.exists():
    with open(config.CANDIDATE_NEOLOGISMS_FILE) as f:
        cands = [l.strip() for l in f if l.strip()]
    logging.info(f"\nFinal candidates ({len(cands)}): {cands[:20]}{'...' if len(cands) > 20 else ''}")

if config.CONTEXT_INDEX_FILE.exists():
    with open(config.CONTEXT_INDEX_FILE) as f:
        ctx = json.load(f)
    logging.info(f"Tokens with context: {len(ctx)}")

elapsed = time.time() - start
logging.info("\n" + "=" * 70)
logging.info(f"GITHUB PIPELINE TEST COMPLETE in {elapsed:.1f}s")
logging.info(f"All outputs in: {TEST_DIR}")
logging.info("=" * 70)
