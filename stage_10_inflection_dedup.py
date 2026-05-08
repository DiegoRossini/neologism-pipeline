#!/usr/bin/env python3

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from config import OUTPUT_DIR, CHECKPOINTS_DIR, LOG_DIR

HAIKU_JUDGE_RESULTS = OUTPUT_DIR / "haiku_4_5_judge_results.tsv"
MAJORITY_VOTE_RESULTS = OUTPUT_DIR / "majority_vote_results.tsv"
DEDUPED_NEOLOGISMS = OUTPUT_DIR / "neologisms_deduplicated.tsv"
ALL_DEDUPED_RESULTS = OUTPUT_DIR / "haiku_4_5_judge_results_dedup.tsv"
INFLECTION_REPORT = OUTPUT_DIR / "neologisms_inflection_report.json"
COMPLETE_FLAG = CHECKPOINTS_DIR / "stage10_dedup_complete.flag"

MIN_BASE_LEN = 3


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "stage_10_inflection_dedup.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def load_final_results(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        if not header_line:
            return rows
        header = header_line.split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append(dict(zip(header, parts)))
    return rows


def candidate_bases(token):
    if token.endswith("ies") and len(token) > MIN_BASE_LEN + 3:
        yield token[:-3] + "y", "ies->y"
    if token.endswith("es") and len(token) > MIN_BASE_LEN + 2:
        yield token[:-2], "es->base"
        yield token[:-1], "es->base+e"
    if token.endswith("ing") and len(token) > MIN_BASE_LEN + 3:
        yield token[:-3], "ing->base"
        yield token[:-3] + "e", "ing->base+e"
    if token.endswith("s") and not token.endswith("ss") and not token.endswith("us") and len(token) > MIN_BASE_LEN + 1:
        yield token[:-1], "s->base"


def find_inflections(neologism_set):
    drop_to_base = {}
    sorted_tokens = sorted(neologism_set, key=lambda t: (-len(t), t))

    for tok in sorted_tokens:
        if tok in drop_to_base:
            continue
        for base, rule in candidate_bases(tok):
            if len(base) < MIN_BASE_LEN:
                continue
            if base in neologism_set and base != tok and base not in drop_to_base:
                drop_to_base[tok] = (base, rule)
                break

    return drop_to_base


def run(args):
    setup_logging()

    if args.input:
        final_results_path = Path(args.input)
    elif HAIKU_JUDGE_RESULTS.exists():
        final_results_path = HAIKU_JUDGE_RESULTS
    elif MAJORITY_VOTE_RESULTS.exists():
        final_results_path = MAJORITY_VOTE_RESULTS
    else:
        final_results_path = HAIKU_JUDGE_RESULTS

    if not final_results_path.exists():
        logging.error(f"Input not found: {final_results_path}")
        logging.error(f"Expected one of: {HAIKU_JUDGE_RESULTS}, {MAJORITY_VOTE_RESULTS}, or --input <path>")
        return False

    logging.info("=" * 70)
    logging.info("STAGE 10: INFLECTIONAL DEDUPLICATION")
    logging.info("=" * 70)
    logging.info(f"Input: {final_results_path}")

    rows = load_final_results(final_results_path)
    logging.info(f"Loaded {len(rows):,} rows")

    label_field = "final_label" if rows and "final_label" in rows[0] else "label"
    logging.info(f"Using label column: {label_field}")

    label_counts = Counter(r.get(label_field, "?") for r in rows)
    logging.info("Label distribution:")
    for label, cnt in label_counts.most_common():
        logging.info(f"  {label}: {cnt:,}")

    neologism_set = {r["token"] for r in rows if r.get(label_field) == "NEOLOGISM"}
    logging.info(f"NEOLOGISM tokens: {len(neologism_set):,}")

    if not neologism_set:
        logging.warning("No NEOLOGISM tokens found; nothing to deduplicate.")
        return False

    drop_to_base = find_inflections(neologism_set)
    logging.info(f"Inflectional variants identified: {len(drop_to_base):,}")

    rule_counts = Counter(rule for _, (_, rule) in drop_to_base.items())
    logging.info("By rule:")
    for rule, cnt in rule_counts.most_common():
        logging.info(f"  {rule}: {cnt:,}")

    deduped_neologisms = neologism_set - set(drop_to_base.keys())
    reduction_pct = 100 * len(drop_to_base) / len(neologism_set) if neologism_set else 0
    logging.info(f"Deduplicated NEOLOGISM count: {len(deduped_neologisms):,} (reduction: {reduction_pct:.1f}%)")

    output_dedup = Path(args.output) if args.output else DEDUPED_NEOLOGISMS
    output_dedup.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dedup, "w", encoding="utf-8") as f:
        f.write("token\n")
        for tok in sorted(deduped_neologisms):
            f.write(f"{tok}\n")
    logging.info(f"Wrote {output_dedup}")

    output_full = Path(args.full_output) if args.full_output else ALL_DEDUPED_RESULTS
    with open(output_full, "w", encoding="utf-8") as f:
        f.write("token\tfinal_label\tsource\tdedup_status\tbase\trule\n")
        for r in rows:
            tok = r["token"]
            label = r.get(label_field, "")
            source = r.get("source", "")
            if label == "NEOLOGISM" and tok in drop_to_base:
                base, rule = drop_to_base[tok]
                f.write(f"{tok}\t{label}\t{source}\tinflection\t{base}\t{rule}\n")
            elif label == "NEOLOGISM":
                f.write(f"{tok}\t{label}\t{source}\tkept\t\t\n")
            else:
                f.write(f"{tok}\t{label}\t{source}\tn_a\t\t\n")
    logging.info(f"Wrote {output_full}")

    output_report = Path(args.report) if args.report else INFLECTION_REPORT
    sample_n = min(300, len(drop_to_base))
    report = {
        "input": str(final_results_path),
        "total_neologisms_input": len(neologism_set),
        "inflections_dropped": len(drop_to_base),
        "deduplicated_count": len(deduped_neologisms),
        "reduction_pct": round(reduction_pct, 2),
        "rule_breakdown": dict(rule_counts),
        "sample_inflections": [
            {"inflection": tok, "base": base, "rule": rule}
            for tok, (base, rule) in sorted(drop_to_base.items())[:sample_n]
        ],
    }
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"Wrote {output_report}")

    if not args.dry_run:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        COMPLETE_FLAG.touch()
        logging.info(f"Marked complete: {COMPLETE_FLAG}")

    logging.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stage 10: Drop inflectional variants from the NEOLOGISM bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reads haiku_4_5_judge_results.tsv (or any TSV with token + final_label/label columns),
filters to NEOLOGISM rows, and removes inflectional variants in favor of the
base form. Past participle (-ed) is always kept because it can function as
an adjective.

Rules (applied in order):
  -ies   ->  -y          (bunnies   -> bunny)
  -es    ->  base or +e  (boxes     -> box;     places   -> place)
  -ing   ->  base or +e  (running   -> run;     dancing  -> dance)
  -s     ->  base        (cats      -> cat;     stans    -> stan)
  -ed                    always kept

A token is dropped only if its derived base form already exists in the
NEOLOGISM set. This avoids false collapses to non-existent bases.

Outputs:
  neologisms_deduplicated.tsv         the final NEOLOGISM list (one per line)
  haiku_4_5_judge_results_dedup.tsv   full results with dedup_status / base / rule
  neologisms_inflection_report.json   stats and a sample of 300 inflection mappings

Examples:
  python stage_10_inflection_dedup.py
  python stage_10_inflection_dedup.py --dry-run
  python stage_10_inflection_dedup.py --input data/output/haiku_4_5_judge_results.tsv
""",
    )
    parser.add_argument("--input", type=str, default=None,
                        help=f"Input TSV. If omitted, prefers haiku_4_5_judge_results.tsv (stage 9 output), "
                             f"falls back to majority_vote_results.tsv (stage 8 output).")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Deduplicated NEOLOGISM list TSV (default: {DEDUPED_NEOLOGISMS})")
    parser.add_argument("--full-output", type=str, default=None,
                        help=f"Full results with dedup info (default: {ALL_DEDUPED_RESULTS})")
    parser.add_argument("--report", type=str, default=None,
                        help=f"JSON report (default: {INFLECTION_REPORT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run end-to-end, write outputs, but don't set complete flag")
    args = parser.parse_args()
    success = run(args)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
