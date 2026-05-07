#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import OUTPUT_DIR, CHECKPOINTS_DIR, CLASSIFICATION_DIR, LOG_DIR

VALID_LABELS = {"ENTITY", "NEOLOGISM", "FOREIGN", "NONE"}

MODEL_KEYS = ["qwen_72b", "llama_70b", "mistral_large"]

DEFAULT_OUTPUT = OUTPUT_DIR / "majority_vote_results.tsv"
DEFAULT_REPORT = CLASSIFICATION_DIR / "majority_vote_report.json"
COMPLETE_FLAG = CHECKPOINTS_DIR / "stage8_majority_vote_complete.flag"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "stage_8_majority_vote.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def load_jsonl_results(path):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tok = rec.get("token")
            label = rec.get("label")
            if tok and label and label in VALID_LABELS:
                results[tok] = label
    return results


def load_tsv_results(path):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("token"):
            raise ValueError(f"Unexpected header in {path}: {header}")
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                logging.warning(f"{path}:{lineno} skipping malformed line: {line.rstrip()}")
                continue
            token, label = parts[0], parts[1].upper().strip()
            if label in VALID_LABELS:
                results[token] = label
    return results


def find_results_for_model(model_key, override=None):
    if override:
        p = Path(override)
        if not p.exists():
            return None
        return p

    jsonl_path = CLASSIFICATION_DIR / f"results_{model_key}.jsonl"
    if jsonl_path.exists():
        return jsonl_path

    tsv_pattern = f"classified_{model_key}_*.tsv"
    tsv_matches = sorted(CLASSIFICATION_DIR.glob(tsv_pattern))
    if tsv_matches:
        return tsv_matches[-1]

    return None


def load_results(path):
    if path.suffix == ".jsonl":
        return load_jsonl_results(path)
    elif path.suffix == ".tsv":
        return load_tsv_results(path)
    raise ValueError(f"Unsupported result file extension: {path.suffix}")


def majority_vote(labels):
    if not labels:
        return "NONE", "no_votes"
    counts = Counter(labels)
    top = counts.most_common()

    n = len(labels)
    if n == 3:
        if top[0][1] == 3:
            return top[0][0], "unanimous"
        if top[0][1] == 2:
            return top[0][0], "majority"
        return "NONE", "tie"
    if n == 2:
        if top[0][1] == 2:
            return top[0][0], "unanimous_2of2"
        return "NONE", "tie_2of2"
    return labels[0], "single"


def is_complete():
    return COMPLETE_FLAG.exists()


def mark_complete():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETE_FLAG.touch()


def run(args):
    setup_logging()

    if is_complete() and not args.force:
        logging.info("Stage 8 already complete. Skipping. Use --force to re-run.")
        return True

    logging.info("=" * 70)
    logging.info("STAGE 8: MAJORITY VOTE")
    logging.info("=" * 70)

    models = {}
    for mk in MODEL_KEYS:
        override = getattr(args, mk, None)
        path = find_results_for_model(mk, override=override)
        if path is None:
            logging.warning(f"No results found for {mk} skipping")
            continue
        models[mk] = load_results(path)
        logging.info(f"  {mk}: {len(models[mk]):,} tokens from {path.name}")

    if len(models) < 2:
        logging.error("Need at least 2 model result sets. Aborting.")
        return False

    all_tokens = set()
    for d in models.values():
        all_tokens.update(d.keys())
    logging.info(f"Total unique tokens across models: {len(all_tokens):,}")

    model_keys_sorted = sorted(models.keys())

    final_results = {}
    rows = []
    vote_stats = Counter()
    label_stats = Counter()
    coverage = Counter()
    per_label_agreement = defaultdict(lambda: Counter())
    disagreements = []

    for token in sorted(all_tokens):
        per_model = {}
        labels = []
        for mk in model_keys_sorted:
            if token in models[mk]:
                per_model[mk] = models[mk][token]
                labels.append(models[mk][token])
            else:
                per_model[mk] = ""

        coverage[len(labels)] += 1
        final_label, vote_type = majority_vote(labels)
        final_results[token] = final_label
        vote_stats[vote_type] += 1
        label_stats[final_label] += 1

        agreement_count = max(Counter(labels).values()) if labels else 0
        per_label_agreement[final_label][vote_type] += 1

        rows.append({
            "token": token,
            "label": final_label,
            "vote_type": vote_type,
            "agreement": agreement_count,
            "per_model": per_model,
        })

        if vote_type in ("majority", "tie", "tie_2of2"):
            disagreements.append({
                "token": token,
                "labels": per_model,
                "final": final_label,
                "vote_type": vote_type,
            })

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_cols = ["token", "label", "vote_type", "agreement"] + model_keys_sorted
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header_cols) + "\n")
        for r in rows:
            cols = [r["token"], r["label"], r["vote_type"], str(r["agreement"])]
            cols += [r["per_model"].get(mk, "") for mk in model_keys_sorted]
            f.write("\t".join(cols) + "\n")
    logging.info(f"Wrote {len(rows):,} rows to {output_path}")

    pairwise = {}
    for i, mk1 in enumerate(model_keys_sorted):
        for mk2 in model_keys_sorted[i + 1:]:
            shared = set(models[mk1].keys()) & set(models[mk2].keys())
            agree = sum(1 for t in shared if models[mk1][t] == models[mk2][t])
            pairwise[f"{mk1}_vs_{mk2}"] = {
                "shared_tokens": len(shared),
                "agree": agree,
                "disagree": len(shared) - agree,
                "agreement_pct": round(100 * agree / len(shared), 2) if shared else 0,
            }

    label_x_vote = {label: dict(per_label_agreement[label]) for label in per_label_agreement}

    report = {
        "models": {mk: len(models[mk]) for mk in model_keys_sorted},
        "total_tokens": len(all_tokens),
        "coverage": {f"{k}_models": v for k, v in sorted(coverage.items())},
        "vote_types": dict(vote_stats.most_common()),
        "final_label_distribution": dict(label_stats.most_common()),
        "label_x_vote_type": label_x_vote,
        "disagreement_count": len(disagreements),
        "disagreement_samples": disagreements[:200],
        "pairwise_agreement": pairwise,
    }

    report_path = Path(args.report) if args.report else DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"Report written to {report_path}")

    logging.info("=" * 70)
    logging.info("MAJORITY VOTE SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Models used: {', '.join(model_keys_sorted)}")
    for mk in model_keys_sorted:
        logging.info(f"  {mk}: {len(models[mk]):,} tokens")
    logging.info(f"Total unique tokens: {len(all_tokens):,}")
    logging.info("Coverage (how many models voted on each token):")
    for k in sorted(coverage):
        logging.info(f"  {k} models: {coverage[k]:,}")
    logging.info("Vote type distribution:")
    for vt, cnt in vote_stats.most_common():
        logging.info(f"  {vt}: {cnt:,}")
    logging.info("Final label distribution:")
    for label, cnt in label_stats.most_common():
        logging.info(f"  {label}: {cnt:,}")
    logging.info(f"Disagreements (non-unanimous): {len(disagreements):,}")
    logging.info("Pairwise agreement:")
    for pair, stats in pairwise.items():
        logging.info(f"  {pair}: {stats['agreement_pct']}% ({stats['agree']:,}/{stats['shared_tokens']:,})")
    logging.info(f"Output TSV: {output_path}")
    logging.info(f"Report:     {report_path}")
    logging.info("=" * 70)

    if not args.dry_run:
        mark_complete()
        logging.info(f"Marked complete: {COMPLETE_FLAG}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stage 8: Majority vote across the 3 LLM classifiers (Qwen, Llama, Mistral)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output TSV columns:
  token        the candidate
  label        consolidated label after voting
  vote_type    unanimous | majority | tie (3-model) ; unanimous_2of2 | tie_2of2 (2-model) ; single
  agreement    number of votes for the winning label (3, 2, or 1)
  qwen_72b     Qwen's label (empty if missing)
  llama_70b    Llama's label
  mistral_large  Mistral's label

The ties default to NONE.

Examples:
  python stage_8_majority_vote.py
  python stage_8_majority_vote.py --force
  python stage_8_majority_vote.py --output my_results.tsv --report my_report.json
""",
    )
    parser.add_argument("--qwen_72b", type=str, help="Override path to Qwen results (.jsonl or .tsv)")
    parser.add_argument("--llama_70b", type=str, help="Override path to Llama results (.jsonl or .tsv)")
    parser.add_argument("--mistral_large", type=str, help="Override path to Mistral results (.jsonl or .tsv)")
    parser.add_argument("--output", type=str, help=f"Output TSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--report", type=str, help=f"Report JSON path (default: {DEFAULT_REPORT})")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete flag exists")
    parser.add_argument("--dry-run", action="store_true", help="Compute and log but do not set complete flag")
    args = parser.parse_args()
    success = run(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
