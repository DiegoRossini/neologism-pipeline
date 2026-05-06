#!/usr/bin/env python3

import os
from pathlib import Path
from multiprocessing import cpu_count

TEST_MODE = False
TEST_FILES_LIMIT = 5

_base_env = os.environ.get("NEOLOGISM_BASE_DIR")
if not _base_env:
    raise RuntimeError(
        "NEOLOGISM_BASE_DIR environment variable is required.\n"
        "Set it to the directory containing your processed corpus subfolders\n"
        "(e.g. processed_comments/ with *.csv.gz files):\n"
        "    export NEOLOGISM_BASE_DIR=/path/to/data_root"
    )
BASE_DIR = Path(_base_env)
SCRIPTS_DIR = Path(__file__).resolve().parent

DATA_DIRS = [
    BASE_DIR / "processed_comments",
]

TEXT_COLUMNS = ["text", "body"]

PIPELINE_DATA_DIR = SCRIPTS_DIR / "data"
TEMP_BATCHES_DIR = PIPELINE_DATA_DIR / "temp_batches"
CHECKPOINTS_DIR = PIPELINE_DATA_DIR / "checkpoints"
OUTPUT_DIR = PIPELINE_DATA_DIR / "output"

TOKEN_COUNTS_FILE = BASE_DIR / "token_counts_final.jsonl"
UNIQUE_TOKENS_FILE = OUTPUT_DIR / "unique_tokens.txt.gz"

CANDIDATE_NEOLOGISMS_FILE = OUTPUT_DIR / "candidate_neologisms.txt"
EXCLUSIONS_DIR = OUTPUT_DIR / "exclusions"

CONTEXT_INDEX_FILE = OUTPUT_DIR / "context_index.json"

CLASSIFICATION_DIR = OUTPUT_DIR / "classification_results"
FILTERED_NEOLOGISMS_FILE = OUTPUT_DIR / "filtered_neologisms.txt"

TOKEN_FREQUENCIES_FILE = OUTPUT_DIR / "token_frequencies.tsv"
UNIQUE_TOKENS_PER_DAY_FILE = OUTPUT_DIR / "unique_tokens_per_day.tsv"
UNIQUE_TOKENS_PER_SUBREDDIT_FILE = OUTPUT_DIR / "unique_tokens_per_subreddit.tsv"
TOKEN_COUNTING_STATS_FILE = OUTPUT_DIR / "token_counting_stats.json"

DUPLICATE_IDS_FILE = OUTPUT_DIR / "duplicate_ids.txt"

VOCAB_DIR = SCRIPTS_DIR / "vocabs"
VOCAB_FILES = [
    VOCAB_DIR / "wikipedia_titles_pre2015_vocab.txt",
    VOCAB_DIR / "wiktionary_pre2015_vocab.txt",
    VOCAB_DIR / "noslang_word_list.txt",
    VOCAB_DIR / "pre2015_vocab.txt",
    VOCAB_DIR / "urban_dict_pre2015_vocab.txt",
    VOCAB_DIR / "wordnet_vocab.txt",
]

TOKENIZATION_CONFIG = {
    "batch_size": 100,
    "n_cores": min(144, cpu_count()),
    "test_mode": False,
    "test_files": 5,
}

VOCAB_FILTERING_CONFIG = {
    "min_token_length": 3,
    "max_token_length": 20,
    "max_edit_distance": 2,
    "min_word_length_typo": 5,
    "min_word_length_segmentation": 6,
}

CONTEXT_CONFIG = {
    "max_contexts_per_token": 10,
    "min_token_count": 15,
    "max_token_count": 100,
    "n_workers": None,
}

PIPELINE_STATE_FILE = OUTPUT_DIR / "pipeline_state.json"

LOG_DIR = OUTPUT_DIR / "logs"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def detect_text_column(filepath):
    import pandas as pd
    header = pd.read_csv(filepath, compression='gzip', nrows=0).columns
    for col in TEXT_COLUMNS:
        if col in header:
            return col
    return None


def get_all_csv_files():
    files = []
    for d in DATA_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.csv.gz")))
    return files


def ensure_directories():
    dirs = [
        PIPELINE_DATA_DIR,
        TEMP_BATCHES_DIR,
        CHECKPOINTS_DIR,
        OUTPUT_DIR,
        CLASSIFICATION_DIR,
        EXCLUSIONS_DIR,
        LOG_DIR,
        VOCAB_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


ensure_directories()
