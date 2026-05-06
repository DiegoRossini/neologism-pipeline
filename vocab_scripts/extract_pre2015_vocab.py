#!/usr/bin/env python3

import argparse
import ast
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CUTOFF_DATE = "2015-01-01"


def setup_logging(output_dir):
    log_file = output_dir / "extract_pre2015_vocab.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def process_single_file(filepath):
    tokens = set()
    stats = {
        "file": str(filepath),
        "total_rows": 0,
        "pre2015_rows": 0,
        "tokens_found": 0,
        "errors": 0,
    }

    try:
        df = pd.read_csv(filepath, compression='gzip', usecols=['date', 'tokenized_text'])
        stats["total_rows"] = len(df)

        for _, row in df.iterrows():
            date_str = row.get('date')
            tokenized_text = row.get('tokenized_text')

            if pd.isna(date_str) or pd.isna(tokenized_text):
                continue

            try:
                if str(date_str) < CUTOFF_DATE:
                    stats["pre2015_rows"] += 1

                    try:
                        tokens_list = ast.literal_eval(tokenized_text)
                        if isinstance(tokens_list, list):
                            for token in tokens_list:
                                if isinstance(token, str) and token.strip():
                                    tokens.add(token.lower().strip())
                    except (ValueError, SyntaxError):
                        stats["errors"] += 1
                        continue
            except Exception:
                stats["errors"] += 1
                continue

        stats["tokens_found"] = len(tokens)

    except Exception as e:
        stats["errors"] += 1
        logging.debug(f"Error processing {filepath}: {e}")

    return tokens, stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocabulary from pre-2015 entries in a tokenized CSV corpus"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=os.environ.get("NEOLOGISM_PRE2015_INPUT_DIR"),
        help="Directory containing tokenized *.csv.gz files with 'date' and 'tokenized_text' columns. "
             "Defaults to env var NEOLOGISM_PRE2015_INPUT_DIR."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("NEOLOGISM_PRE2015_OUTPUT_DIR"),
        help="Output directory for the vocabulary file and stats. "
             "Defaults to env var NEOLOGISM_PRE2015_OUTPUT_DIR."
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: all CPUs)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of files to process (for testing)"
    )
    args = parser.parse_args()

    if not args.input_dir:
        parser.error(
            "--input-dir not provided and NEOLOGISM_PRE2015_INPUT_DIR is not set. "
            "Provide a directory of tokenized *.csv.gz files."
        )
    if not args.output_dir:
        parser.error(
            "--output-dir not provided and NEOLOGISM_PRE2015_OUTPUT_DIR is not set."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logging.info("=" * 70)
    logging.info("EXTRACT PRE-2015 VOCABULARY")
    logging.info("=" * 70)
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Cutoff date: {CUTOFF_DATE} (exclusive)")

    csv_files = sorted(input_dir.glob("*.csv.gz"))

    if args.limit:
        csv_files = csv_files[:args.limit]
        logging.info(f"LIMIT MODE: Processing only {len(csv_files)} files")
    else:
        logging.info(f"Found {len(csv_files):,} CSV files to process")

    if not csv_files:
        logging.error("No CSV files found!")
        return 1

    n_workers = args.n_workers or os.cpu_count()
    logging.info(f"Using {n_workers} parallel workers")

    all_tokens = set()
    total_stats = {
        "total_files": len(csv_files),
        "files_with_pre2015": 0,
        "total_rows": 0,
        "pre2015_rows": 0,
        "total_errors": 0,
    }

    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_file, f): f for f in csv_files}

        with tqdm(total=len(csv_files), desc="Processing files") as pbar:
            for future in as_completed(futures):
                try:
                    tokens, stats = future.result()

                    if tokens:
                        all_tokens.update(tokens)
                        total_stats["files_with_pre2015"] += 1

                    total_stats["total_rows"] += stats["total_rows"]
                    total_stats["pre2015_rows"] += stats["pre2015_rows"]
                    total_stats["total_errors"] += stats["errors"]

                except Exception as e:
                    logging.error(f"Error in future: {e}")
                    total_stats["total_errors"] += 1

                pbar.update(1)
                pbar.set_postfix({
                    "tokens": len(all_tokens),
                    "pre2015_rows": total_stats["pre2015_rows"]
                })

    elapsed = datetime.now() - start_time

    logging.info(f"Processing complete in {elapsed}")
    logging.info(f"Total unique tokens from pre-2015: {len(all_tokens):,}")

    vocab_file = output_dir / "pre2015_vocab.txt"
    logging.info(f"Saving vocabulary to {vocab_file}...")

    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in sorted(all_tokens):
            f.write(f"{token}\n")

    total_stats["unique_tokens"] = len(all_tokens)
    total_stats["processing_time_seconds"] = elapsed.total_seconds()
    total_stats["cutoff_date"] = CUTOFF_DATE
    total_stats["timestamp"] = datetime.now().isoformat()

    stats_file = output_dir / "pre2015_vocab_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(total_stats, f, indent=2)

    logging.info("=" * 70)
    logging.info("EXTRACTION COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"Files processed:     {total_stats['total_files']:,}")
    logging.info(f"Files with pre-2015: {total_stats['files_with_pre2015']:,}")
    logging.info(f"Total rows scanned:  {total_stats['total_rows']:,}")
    logging.info(f"Pre-2015 rows:       {total_stats['pre2015_rows']:,}")
    logging.info(f"Unique tokens:       {len(all_tokens):,}")
    logging.info(f"Errors:              {total_stats['total_errors']:,}")
    logging.info(f"Output vocab:        {vocab_file}")
    logging.info(f"Output stats:        {stats_file}")
    logging.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
