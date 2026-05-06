#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import (
    CHECKPOINTS_DIR,
    CANDIDATE_NEOLOGISMS_FILE,
    CONTEXT_INDEX_FILE,
    CLASSIFICATION_DIR,
)

from stage_7_llm_classify import (
    create_llm_prompt,
    create_single_token_prompt,
    parse_llm_response,
    select_diverse_contexts,
    detect_foreign_by_context,
    load_duplicate_ids,
    load_checkpoint,
    save_checkpoint,
    load_tokens,
    load_context_index,
)

from config import DUPLICATE_IDS_FILE

DEFAULT_MODEL = "claude-3-5-haiku-20241022"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_USE_BATCH_API = True
MODEL_KEY = "claude"

def setup_logging():
    log_file = Path(__file__).parent / "stage_8_claude_classify.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    print(f"Logging to: {log_file}")

def get_anthropic_client():
    try:
        from anthropic import Anthropic
    except ImportError:
        logging.error("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    return Anthropic(api_key=api_key)

def classify_realtime(client, model, prompt, expected_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            results = parse_llm_response(response_text, expected_tokens)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            logging.debug(f"[Real-time] Input: {input_tokens}, Output: {output_tokens}")

            return results, response_text, None

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {t: "ERROR" for t in expected_tokens}, "", str(e)


def create_batch_requests(groups, model, tokens_per_prompt):
    requests = []

    for group_idx, group in enumerate(groups):
        if tokens_per_prompt > 1:
            prompt = create_llm_prompt(group)
        else:
            prompt = create_single_token_prompt(group[0])

        expected_tokens = [item["token"] for item in group]

        request = {
            "custom_id": f"group_{group_idx}",
            "params": {
                "model": model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}]
            }
        }
        requests.append({
            "request": request,
            "tokens": expected_tokens,
        })

    return requests

def submit_batch_job(client, requests, model):
    batch_file = CLASSIFICATION_DIR / "batch_requests_claude.jsonl"
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    with open(batch_file, 'w') as f:
        for req_data in requests:
            f.write(json.dumps(req_data["request"]) + '\n')

    logging.info(f"Created batch file with {len(requests)} requests")

    try:
        batch_requests_list = []
        with open(batch_file, 'r') as f:
            for line in f:
                if line.strip():
                    batch_requests_list.append(json.loads(line))

        batch = client.messages.batches.create(
            requests=batch_requests_list
        )

        logging.info(f"Batch job submitted: {batch.id}")
        logging.info(f"Status: {batch.processing_status}")

        metadata = {
            "batch_id": batch.id,
            "created_at": str(batch.created_at),
            "model": model,
            "request_counts": batch.request_counts.__dict__ if hasattr(batch, 'request_counts') else {},
            "tokens_map": {req_data["request"]["custom_id"]: req_data["tokens"] for req_data in requests},
        }

        metadata_file = CLASSIFICATION_DIR / f"batch_metadata_{batch.id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logging.info(f"Batch metadata saved to {metadata_file}")
        return batch.id, metadata_file

    except Exception as e:
        logging.error(f"Failed to submit batch: {e}")
        raise

def check_batch_status(client, batch_id):
    try:
        batch = client.messages.batches.retrieve(batch_id)
        return batch
    except Exception as e:
        logging.error(f"Failed to check batch status: {e}")
        raise

def retrieve_batch_results(client, batch_id, metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    tokens_map = metadata["tokens_map"]

    try:
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            logging.warning(f"Batch not complete. Status: {batch.processing_status}")
            return None

        results_iter = client.messages.batches.results(batch_id)

        all_results = {}
        total_input_tokens = 0
        total_output_tokens = 0

        raw_responses_path = CLASSIFICATION_DIR / f"raw_responses_{MODEL_KEY}_batch.jsonl"
        raw_responses_file = open(raw_responses_path, 'a', encoding='utf-8')

        for result in results_iter:
            custom_id = result.custom_id
            expected_tokens = tokens_map[custom_id]

            if result.result.type == "succeeded":
                response = result.result.message
                response_text = response.content[0].text

                raw_responses_file.write(json.dumps({
                    "custom_id": custom_id,
                    "tokens": expected_tokens,
                    "response": response_text,
                }, ensure_ascii=False) + '\n')

                batch_results = parse_llm_response(response_text, expected_tokens)
                all_results.update(batch_results)

                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

            elif result.result.type == "errored":
                for token in expected_tokens:
                    all_results[token] = "ERROR"
                logging.warning(f"Batch {custom_id} failed: {result.result.error}")

        raw_responses_file.close()
        logging.info(f"[Batch] Raw responses saved to {raw_responses_path}")

        if "haiku" in metadata["model"].lower():
            input_cost = total_input_tokens * 0.50 / 1_000_000
            output_cost = total_output_tokens * 2.50 / 1_000_000
        elif "sonnet" in metadata["model"].lower():
            input_cost = total_input_tokens * 1.50 / 1_000_000
            output_cost = total_output_tokens * 7.50 / 1_000_000
        else:
            input_cost = total_input_tokens * 7.50 / 1_000_000
            output_cost = total_output_tokens * 37.50 / 1_000_000

        total_cost = input_cost + output_cost

        logging.info(f"[Batch API] Total tokens - Input: {total_input_tokens:,}, Output: {total_output_tokens:,}")
        logging.info(f"[Batch API] Total cost: ${total_cost:.4f} (50% savings vs real-time)")

        return all_results, (total_input_tokens, total_output_tokens, total_cost)

    except Exception as e:
        logging.error(f"Failed to retrieve batch results: {e}")
        raise

def is_complete():
    checkpoint_file = CHECKPOINTS_DIR / "stage8_complete.flag"
    return checkpoint_file.exists()

def mark_complete():
    checkpoint_file = CHECKPOINTS_DIR / "stage8_complete.flag"
    checkpoint_file.touch()

def run(
    force=False,
    model=None,
    batch_size=None,
    max_retries=None,
    dry_run=False,
    use_batch_api=None,
    batch_id=None,
    tokens_per_prompt=10,
    max_tokens=None,
):
    setup_logging()

    if is_complete() and not force:
        logging.info("Stage 8 already complete. Skipping. Use force=True to re-run.")
        return True

    logging.info("=" * 70)
    logging.info("STAGE 8: CLAUDE NER CLASSIFICATION")
    logging.info("=" * 70)

    if not CANDIDATE_NEOLOGISMS_FILE.exists():
        logging.error(f"Prerequisite not found: {CANDIDATE_NEOLOGISMS_FILE}")
        return False

    if not CONTEXT_INDEX_FILE.exists():
        logging.error(f"Prerequisite not found: {CONTEXT_INDEX_FILE}")
        return False

    model = model or DEFAULT_MODEL
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    max_retries = max_retries or DEFAULT_MAX_RETRIES
    use_batch_api = use_batch_api if use_batch_api is not None else DEFAULT_USE_BATCH_API

    logging.info(f"Model: {model}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Tokens per prompt: {tokens_per_prompt}")
    logging.info(f"Max retries: {max_retries}")
    logging.info(f"API mode: {'Batch API (50% savings)' if use_batch_api else 'Real-time API'}")

    client = get_anthropic_client()
    logging.info("Anthropic client initialized")

    if batch_id:
        logging.info(f"Retrieving batch results: {batch_id}")
        metadata_file = CLASSIFICATION_DIR / f"batch_metadata_{batch_id}.json"
        if not metadata_file.exists():
            metadata_file = CLASSIFICATION_DIR / "batch_metadata_combined.json"
        if not metadata_file.exists():
            logging.error(f"Metadata file not found for batch {batch_id}")
            return False

        batch_result = retrieve_batch_results(client, batch_id, metadata_file)
        if batch_result is None:
            batch = check_batch_status(client, batch_id)
            logging.info(f"Batch status: {batch.processing_status}")
            logging.info(f"Request counts: {batch.request_counts}")
            logging.info("Batch not ready yet. Check back later.")
            return False

        results, (input_tokens, output_tokens, total_cost) = batch_result

        llm_results = load_checkpoint(MODEL_KEY)
        llm_results.update(results)
        save_checkpoint(MODEL_KEY, llm_results)

        logging.info(f"Batch results saved to checkpoint: {len(results)} tokens classified")
        logging.info(f"Total in checkpoint: {len(llm_results)}")
        return True

    logging.info("Loading tokens...")
    all_tokens = load_tokens(CANDIDATE_NEOLOGISMS_FILE)
    logging.info(f"Loaded {len(all_tokens):,} total tokens")

    llm_results = load_checkpoint(MODEL_KEY)
    already_done = set(llm_results.keys())

    remaining_tokens = [t for t in all_tokens if t not in already_done]

    if dry_run:
        remaining_tokens = remaining_tokens[:100]
        logging.info(f"DRY RUN: Processing only first 100 remaining tokens")

    if max_tokens is not None:
        remaining_tokens = remaining_tokens[:max_tokens]
        logging.info(f"--max-tokens: Processing {len(remaining_tokens)} tokens this run")

    logging.info(f"[{MODEL_KEY}] Total tokens: {len(all_tokens)}, "
                 f"already done: {len(already_done)}, remaining: {len(remaining_tokens)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not remaining_tokens:
        logging.info(f"[{MODEL_KEY}] All tokens already classified from checkpoint")
        tokens_with_info = []
        remaining = []
    else:
        logging.info("Loading context index...")
        context_index = load_context_index(CONTEXT_INDEX_FILE)

        duplicate_ids = load_duplicate_ids(DUPLICATE_IDS_FILE)

        logging.info("=== Preparing contexts with deduplication and diversity selection ===")

        tokens_with_info = []
        context_stats_log = []

        for token in tqdm(remaining_tokens, desc="Selecting contexts"):
            raw_contexts = context_index.get(token, [])

            contexts_with_metadata = []
            for ctx in raw_contexts:
                if isinstance(ctx, dict):
                    contexts_with_metadata.append(ctx)
                else:
                    contexts_with_metadata.append({
                        'text': ctx,
                        'subreddit': 'unknown',
                        'post_id': '',
                    })

            selected_contexts, stats = select_diverse_contexts(
                contexts_with_metadata, token, duplicate_ids
            )

            tokens_with_info.append({
                "token": token,
                "contexts": selected_contexts,
            })

            context_stats_log.append({
                "token": token,
                "raw": stats['raw_contexts'],
                "after_dedup": stats['after_dedup'],
                "used": stats['examples_used'],
                "subreddits": stats['subreddits_used'],
            })

        total_raw = sum(s['raw'] for s in context_stats_log)
        total_dedup = sum(s['after_dedup'] for s in context_stats_log)
        total_used = sum(s['used'] for s in context_stats_log)

        logging.info(f"[Context Selection] Raw contexts: {total_raw:,}")
        logging.info(f"[Context Selection] After deduplication: {total_dedup:,}")
        logging.info(f"[Context Selection] Used in prompts: {total_used:,}")
        if remaining_tokens:
            logging.info(f"[Context Selection] Avg contexts per token: {total_used / len(remaining_tokens):.1f}")

        tokens_with_info, foreign_detected = detect_foreign_by_context(tokens_with_info)
        if foreign_detected:
            logging.info(f"[Foreign Detection] Removed {len(foreign_detected)} foreign tokens before classification")

        remaining = tokens_with_info

    logging.info(f"[{MODEL_KEY}] {len(remaining)} tokens to process, "
                 f"tokens_per_prompt: {tokens_per_prompt}")

    if not remaining:
        logging.info(f"[{MODEL_KEY}] All tokens already classified from checkpoint")
    else:
        groups = []
        for i in range(0, len(remaining), tokens_per_prompt):
            groups.append(remaining[i:i + tokens_per_prompt])

        logging.info(f"[{MODEL_KEY}] {len(remaining)} tokens → {len(groups)} prompt groups "
                     f"(~{tokens_per_prompt} tokens/prompt)")

        if use_batch_api:
            logging.info("Using Batch API (50% cost savings)")

            batch_requests = create_batch_requests(groups, model, tokens_per_prompt)
            logging.info(f"Created {len(batch_requests)} batch requests")

            batch_id_new, metadata_file = submit_batch_job(client, batch_requests, model)

            logging.info("")
            logging.info("=" * 70)
            logging.info("BATCH JOB SUBMITTED SUCCESSFULLY!")
            logging.info(f"Batch ID: {batch_id_new}")
            logging.info(f"Metadata: {metadata_file}")
            logging.info("")
            logging.info("The batch will be processed within 24 hours.")
            logging.info("")
            logging.info("To check status and retrieve results:")
            logging.info(f"  python {__file__} --batch-id {batch_id_new}")
            logging.info("=" * 70)
            return True

        else:
            logging.info("Using Real-time API")

            start_time = time.time()
            num_groups = len(groups)

            raw_responses_path = CLASSIFICATION_DIR / f"raw_responses_{MODEL_KEY}.jsonl"
            raw_responses_file = open(raw_responses_path, 'a', encoding='utf-8')
            logging.info(f"[{MODEL_KEY}] Saving raw responses to {raw_responses_path}")

            for group_idx in tqdm(range(num_groups), desc=f"{MODEL_KEY} classifying"):
                group = groups[group_idx]
                expected_tokens = [item["token"] for item in group]

                if tokens_per_prompt > 1:
                    prompt = create_llm_prompt(group)
                else:
                    prompt = create_single_token_prompt(group[0])

                results, response_text, error = classify_realtime(
                    client, model, prompt, expected_tokens, max_retries
                )

                raw_responses_file.write(json.dumps({
                    "group_idx": group_idx,
                    "tokens": expected_tokens,
                    "response": response_text,
                }, ensure_ascii=False) + '\n')
                raw_responses_file.flush()

                if error:
                    logging.warning(f"[{MODEL_KEY}] Group {group_idx} error: {error}")
                else:
                    llm_results.update(results)

                if (group_idx + 1) % 50 == 0:
                    save_checkpoint(MODEL_KEY, llm_results)
                    logging.info(f"[{MODEL_KEY}] Checkpoint saved at group {group_idx + 1}/{num_groups}")

                if group_idx < num_groups - 1:
                    time.sleep(0.3)

            raw_responses_file.close()
            logging.info(f"[{MODEL_KEY}] Raw responses saved to {raw_responses_path}")

            save_checkpoint(MODEL_KEY, llm_results)

            elapsed = time.time() - start_time
            logging.info(f"[{MODEL_KEY}] Primary pass completed in {elapsed:.1f}s")

            failed_tokens = [t for t, v in llm_results.items() if v in ["ERROR", "UNKNOWN"]]

            if failed_tokens:
                logging.info(f"[{MODEL_KEY}] Retrying {len(failed_tokens)} ERROR/UNKNOWN tokens (single-token mode)...")
                token_info_map = {t["token"]: t for t in tokens_with_info}
                retry_items = [token_info_map[t] for t in failed_tokens if t in token_info_map]

                retry_responses_path = CLASSIFICATION_DIR / f"raw_responses_{MODEL_KEY}_retry.jsonl"
                retry_responses_file = open(retry_responses_path, 'a', encoding='utf-8')

                for retry_idx, item in enumerate(tqdm(retry_items, desc=f"{MODEL_KEY} retry")):
                    token = item["token"]
                    prompt = create_single_token_prompt(item)

                    results, response_text, error = classify_realtime(
                        client, model, prompt, [token], max_retries
                    )

                    retry_responses_file.write(json.dumps({
                        "retry_idx": retry_idx,
                        "tokens": [token],
                        "response": response_text,
                    }, ensure_ascii=False) + '\n')
                    retry_responses_file.flush()

                    for t, label in results.items():
                        if label not in ["ERROR", "UNKNOWN"]:
                            llm_results[t] = label

                    time.sleep(0.3)

                retry_responses_file.close()
                save_checkpoint(MODEL_KEY, llm_results)

    final_unknown = sum(1 for v in llm_results.values() if v in ["ERROR", "UNKNOWN"])
    if final_unknown > 0:
        logging.info(f"[{MODEL_KEY}] Defaulting {final_unknown} unresolved tokens to NONE")
        for token in llm_results:
            if llm_results[token] in ["ERROR", "UNKNOWN"]:
                llm_results[token] = "NONE"

    final_stats = {
        "entity": sum(1 for v in llm_results.values() if v == "ENTITY"),
        "neologism": sum(1 for v in llm_results.values() if v == "NEOLOGISM"),
        "foreign": sum(1 for v in llm_results.values() if v == "FOREIGN"),
        "none": sum(1 for v in llm_results.values() if v == "NONE"),
    }

    logging.info(f"[{MODEL_KEY}] Final: {final_stats['entity']} ENTITY | "
                 f"{final_stats['neologism']} NEOLOGISM | "
                 f"{final_stats['foreign']} FOREIGN | "
                 f"{final_stats['none']} NONE")

    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    tsv_path = CLASSIFICATION_DIR / f"classified_{MODEL_KEY}_{timestamp}.tsv"
    output_tokens = sorted(llm_results.keys())

    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("token\tlabel\tis_entity\n")
        for token in output_tokens:
            label = llm_results.get(token, "NONE")
            is_entity = "true" if label == "ENTITY" else "false"
            f.write(f"{token}\t{label}\t{is_entity}\n")

    logging.info(f"[{MODEL_KEY}] TSV saved to {tsv_path}")

    report_path = CLASSIFICATION_DIR / f"report_{MODEL_KEY}_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "model_key": MODEL_KEY,
        "model_name": model,
        "total_tokens": len(output_tokens),
        "results": final_stats,
        "all_results": llm_results,
        "entities_list": [t for t in output_tokens if llm_results.get(t) == "ENTITY"],
        "neologisms_list": [t for t in output_tokens if llm_results.get(t) == "NEOLOGISM"],
        "foreign_list": [t for t in output_tokens if llm_results.get(t) == "FOREIGN"],
        "none_list": [t for t in output_tokens if llm_results.get(t) == "NONE"],
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info(f"[{MODEL_KEY}] Report saved to {report_path}")

    if not dry_run and max_tokens is None:
        mark_complete()

    logging.info("=" * 70)
    logging.info("STAGE 8 COMPLETE!")
    logging.info(f"Model: {model}")
    logging.info(f"Total tokens processed: {len(output_tokens):,}")
    logging.info(f"  - Entities: {final_stats['entity']:,}")
    logging.info(f"  - Neologisms: {final_stats['neologism']:,}")
    logging.info(f"  - Foreign: {final_stats['foreign']:,}")
    logging.info(f"  - NONE: {final_stats['none']:,}")
    logging.info(f"Output: {tsv_path}")
    logging.info("=" * 70)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 8: Claude Classification (same prompts/parsing as stage 7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uses the EXACT same prompts, context selection, and parsing as stage_7_llm_classify.py.
Only difference: calls Claude API instead of local model.generate().

Examples:
  python stage_8_claude_classify.py

  python stage_8_claude_classify.py --batch-id msgbatch_abc123

  python stage_8_claude_classify.py --no-batch-api

  python stage_8_claude_classify.py --no-batch-api --tokens-per-prompt 1

  python stage_8_claude_classify.py --dry-run

  python stage_8_claude_classify.py --max-tokens 35000

  python stage_8_claude_classify.py --force

Environment:
  ANTHROPIC_API_KEY must be set
        """
    )

    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Number of prompt groups per checkpoint cycle (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--tokens-per-prompt", type=int, default=10,
        help="Tokens per multi-token prompt (default: 10, same as stage 7)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per API call (default: {DEFAULT_MAX_RETRIES})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only first 100 tokens (uses real-time API)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if complete"
    )
    parser.add_argument(
        "--no-batch-api", dest="use_batch_api", action="store_false",
        help="Use real-time API instead of Batch API (2x cost)"
    )
    parser.add_argument(
        "--batch-id", type=str, default=None,
        help="Resume from existing batch job (provide batch_id)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Limit to first N tokens (e.g. --max-tokens 35000)"
    )

    args = parser.parse_args()

    use_batch_api = args.use_batch_api if not args.dry_run else False

    success = run(
        force=args.force,
        model=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        use_batch_api=use_batch_api,
        batch_id=args.batch_id,
        tokens_per_prompt=args.tokens_per_prompt,
        max_tokens=args.max_tokens,
    )

    sys.exit(0 if success else 1)
