#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import OUTPUT_DIR, CLASSIFICATION_DIR

VALID_LABELS = {"ENTITY", "NEOLOGISM", "FOREIGN", "NONE"}

MODEL_KEYS = ["qwen_72b", "llama_70b", "mistral_large"]

DEFAULT_OUTPUT = OUTPUT_DIR / "FINAL_RESULTS.tsv"
DEFAULT_REPORT = CLASSIFICATION_DIR / "majority_vote_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_latest_result(model_key):
    pattern = f"classified_{model_key}_*.tsv"
    matches = sorted(CLASSIFICATION_DIR.glob(pattern))
    if matches:
        return matches[-1]
    return None


def load_tsv(path):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("token"):
            raise ValueError(f"Unexpected header in {path}: {header}")
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                logger.warning(f"{path}:{lineno} — skipping malformed line: {line.rstrip()}")
                continue
            token, label = parts[0], parts[1].upper().strip()
            if label not in VALID_LABELS:
                label = "NONE"
            results[token] = label
    return results


def majority_vote(labels):
    counts = Counter(labels)
    top = counts.most_common()

    if len(labels) == 3:
        if top[0][1] == 3:
            return top[0][0], "unanimous"
        elif top[0][1] == 2:
            return top[0][0], "majority"
        else:
            return "NONE", "tie"
    elif len(labels) == 2:
        if top[0][1] == 2:
            return top[0][0], "unanimous_2"
        else:
            return "NONE", "tie_2"
    elif len(labels) == 1:
        return labels[0], "single"
    else:
        return "NONE", "no_votes"


def run(args):
    logger.info("Discovering model results...")
    models = {}

    for model_key in MODEL_KEYS:
        override = getattr(args, model_key, None) if args else None
        if override:
            path = Path(override)
        else:
            path = find_latest_result(model_key)

        if path is None or not path.exists():
            logger.warning(f"No results found for {model_key} — skipping")
            continue

        models[model_key] = load_tsv(path)
        logger.info(f"  {model_key}: {len(models[model_key]):,} tokens from {path.name}")

    if len(models) < 2:
        logger.error("Need at least 2 model result files. Aborting.")
        sys.exit(1)

    all_tokens = set()
    for d in models.values():
        all_tokens.update(d.keys())
    logger.info(f"Total unique tokens across models: {len(all_tokens):,}")

    final_results = {}
    vote_stats = Counter()
    label_stats = Counter()
    coverage = Counter()
    per_label_agreement = defaultdict(lambda: {"unanimous": 0, "majority": 0, "tie": 0})
    disagreements = []

    model_keys = sorted(models.keys())

    for token in sorted(all_tokens):
        labels = []
        per_model = {}
        for mk in model_keys:
            if token in models[mk]:
                labels.append(models[mk][token])
                per_model[mk] = models[mk][token]

        coverage[len(labels)] += 1
        final_label, vote_type = majority_vote(labels)
        final_results[token] = final_label
        vote_stats[vote_type] += 1
        label_stats[final_label] += 1

        base_type = vote_type.replace("_2", "")
        if base_type in ("unanimous", "single"):
            base_type = "unanimous"
        per_label_agreement[final_label][base_type] += 1

        if vote_type in ("majority", "tie", "tie_2"):
            disagreements.append({
                "token": token,
                "labels": per_model,
                "final": final_label,
                "vote_type": vote_type,
            })

    output_path = Path(args.output) if args and args.output else DEFAULT_OUTPUT
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("token\tlabel\n")
        for token in sorted(final_results):
            f.write(f"{token}\t{final_results[token]}\n")
    logger.info(f"Written {len(final_results):,} tokens to {output_path}")

    report = {
        "models": {mk: len(models[mk]) for mk in model_keys},
        "total_tokens": len(all_tokens),
        "coverage": {f"{k}_models": v for k, v in sorted(coverage.items())},
        "vote_types": dict(vote_stats.most_common()),
        "final_label_distribution": dict(label_stats.most_common()),
        "disagreement_count": len(disagreements),
        "disagreement_samples": disagreements[:100],
    }

    report["per_label_agreement"] = dict(per_label_agreement)

    pairwise = {}
    for i, mk1 in enumerate(model_keys):
        for mk2 in model_keys[i + 1:]:
            shared = set(models[mk1].keys()) & set(models[mk2].keys())
            agree = sum(1 for t in shared if models[mk1][t] == models[mk2][t])
            pairwise[f"{mk1}_vs_{mk2}"] = {
                "shared_tokens": len(shared),
                "agree": agree,
                "disagree": len(shared) - agree,
                "agreement_pct": round(100 * agree / len(shared), 2) if shared else 0,
            }
    report["pairwise_agreement"] = pairwise

    report_path = Path(args.report) if args and args.report else DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report written to {report_path}")

    print("\n" + "=" * 60)
    print("MAJORITY VOTE SUMMARY")
    print("=" * 60)
    print(f"\nModels: {', '.join(model_keys)}")
    for mk in model_keys:
        print(f"  {mk}: {len(models[mk]):,} tokens")
    print(f"\nTotal tokens: {len(all_tokens):,}")
    print(f"\nCoverage:")
    for k in sorted(coverage):
        print(f"  {k} models: {coverage[k]:,}")
    print(f"\nVote types:")
    for vt, cnt in vote_stats.most_common():
        print(f"  {vt}: {cnt:,}")
    print(f"\nFinal label distribution:")
    for label, cnt in label_stats.most_common():
        print(f"  {label}: {cnt:,}")
    print(f"\nDisagreements: {len(disagreements):,}")
    print(f"\nPairwise agreement:")
    for pair, stats in pairwise.items():
        print(f"  {pair}: {stats['agreement_pct']}% ({stats['agree']:,}/{stats['shared_tokens']:,})")
    print(f"\nOutput: {output_path}")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 8: Majority vote across LLM classifications")
    parser.add_argument("--qwen_72b", type=str, help="Path to Qwen results TSV (overrides auto-discovery)")
    parser.add_argument("--llama_70b", type=str, help="Path to Llama results TSV (overrides auto-discovery)")
    parser.add_argument("--mistral_large", type=str, help="Path to Mistral results TSV (overrides auto-discovery)")
    parser.add_argument("--output", type=str, help="Path for output TSV")
    parser.add_argument("--report", type=str, help="Path for report JSON")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
