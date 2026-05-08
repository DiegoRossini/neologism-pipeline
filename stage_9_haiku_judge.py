#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR, CHECKPOINTS_DIR, CLASSIFICATION_DIR, LOG_DIR

VALID_LABELS = {"ENTITY", "NEOLOGISM", "FOREIGN", "NONE"}

DEFAULT_MODEL = "claude-haiku-4-5"
MAX_NEW_TOKENS = 32
POLL_INTERVAL_SEC = 30
BATCH_CHUNK_SIZE = 2000
BATCH_TIMEOUT_SEC = 1800

MAJORITY_VOTE_RESULTS = OUTPUT_DIR / "majority_vote_results.tsv"
CANDIDATES_PRE_LLM = OUTPUT_DIR / "stage7_candidates_pre_llm.jsonl"
HAIKU_RESULTS = CLASSIFICATION_DIR / "results_haiku.jsonl"
HAIKU_RAW_RESPONSES = CLASSIFICATION_DIR / "raw_responses_haiku.jsonl"
HAIKU_JUDGE_RESULTS = OUTPUT_DIR / "haiku_4_5_judge_results.tsv"
COMPLETE_FLAG = CHECKPOINTS_DIR / "stage9_haiku_complete.flag"
BATCH_STATE_FILE = CHECKPOINTS_DIR / "stage9_haiku_batch.json"

SYSTEM_PROMPT = """You are a linguistic judge classifying English tokens from Reddit discourse.

Each prompt gives you ONE token and up to 3 example contexts where it appears. Classify the token into ONE category:

ENTITY - Proper nouns: people, fictional characters, brands, products, companies, games, movies, places, apps. Examples: elon, fortnite, reddit, tokyo, pikachu, iphone.

NEOLOGISM - New English words coined since approximately 2015, slang, OR words derived from proper nouns. Examples: doomscrolling, ghosting, rizz, bussin, adulting, covidiot, youtuber, redditor, trumpian, instagrammable, uberize, googlable.

FOREIGN - Non-English words. Examples: alem, anspielung, yapmyorum, ademas.

NONE - Reddit usernames, typos, programming identifiers, gibberish, unclear words, random character sequences.

Critical rules:
1. Words derived from proper nouns are NEOLOGISM, not ENTITY (youtuber -> NEOLOGISM, youtube -> ENTITY).
2. When uncertain, choose NONE.
3. Use the context and subreddit to understand actual usage.

Reply with EXACTLY one line in this format: TOKEN:LABEL
No explanations, no preamble, no extra text."""


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "stage_9_haiku_judge.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def get_anthropic_client():
    try:
        import anthropic
    except ImportError:
        logging.error("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def load_majority_vote_results(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        if not header_line:
            return rows
        header = header_line.split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            row = dict(zip(header, parts))
            rows.append(row)
    return rows


def load_candidate_contexts(path, tokens_set):
    found = {}
    target = set(tokens_set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tok = rec.get("token")
            if tok in target:
                found[tok] = rec.get("contexts", [])
                if len(found) == len(target):
                    break
    return found


def build_user_prompt(token, contexts):
    lines = [f"TOKEN: {token}"]
    if contexts:
        for i, ctx in enumerate(contexts, 1):
            text = ctx.get("text", "No context")
            sub = ctx.get("subreddit", "unknown")
            lines.append(f'  context_{i} (r/{sub}): "{text}"')
    else:
        lines.append("  context: No context available")
    lines.append("")
    lines.append(f"Answer: {token}:LABEL")
    return "\n".join(lines)


def parse_label(response_text):
    if not response_text:
        return "UNKNOWN"
    text = response_text.strip()
    for line in text.split("\n"):
        line = line.strip()
        if ":" in line:
            parts = line.rsplit(":", 1)
            if len(parts) == 2:
                cand = parts[1].strip().upper()
                if cand in VALID_LABELS:
                    return cand
                for valid in VALID_LABELS:
                    if valid in cand:
                        return valid
    upper = text.upper()
    for valid in VALID_LABELS:
        if valid in upper:
            return valid
    return "UNKNOWN"


def load_done_haiku_tokens(path):
    done = {}
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tok = rec.get("token")
            label = rec.get("label")
            if tok and label and label in VALID_LABELS:
                done[tok] = label
    return done


def call_realtime_one(client, model, token, prompt):
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        label = parse_label(text)
        if label == "UNKNOWN":
            label = "NONE"
        usage = resp.usage
        return {
            "token": token,
            "label": label,
            "response": text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_create": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "error": None,
        }
    except Exception as e:
        return {
            "token": token,
            "label": "NONE",
            "response": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_create": 0,
            "error": str(e),
        }


def run_realtime(client, model, requests_with_tokens, concurrency):
    HAIKU_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results_f = open(HAIKU_RESULTS, "a", encoding="utf-8")
    raw_f = open(HAIKU_RAW_RESPONSES, "a", encoding="utf-8")
    lock = threading.Lock()
    n_done = 0
    n_total = len(requests_with_tokens)
    in_tok = out_tok = cache_read = cache_create = n_errored = 0
    start = time.time()

    def write_result(r):
        nonlocal n_done, in_tok, out_tok, cache_read, cache_create, n_errored
        with lock:
            results_f.write(json.dumps({"token": r["token"], "label": r["label"]}, ensure_ascii=False) + "\n")
            if r["error"]:
                raw_f.write(json.dumps({"token": r["token"], "error": r["error"]}, ensure_ascii=False) + "\n")
                n_errored += 1
            else:
                raw_f.write(json.dumps({"token": r["token"], "response": r["response"]}, ensure_ascii=False) + "\n")
            results_f.flush()
            raw_f.flush()
            in_tok += r["input_tokens"]
            out_tok += r["output_tokens"]
            cache_read += r["cache_read"]
            cache_create += r["cache_create"]
            n_done += 1
            if n_done % 50 == 0 or n_done == n_total:
                elapsed = time.time() - start
                rate = n_done / elapsed if elapsed > 0 else 0
                logging.info(f"  realtime progress: {n_done}/{n_total} ({rate:.1f} tok/s) errored={n_errored} input={in_tok:,} output={out_tok:,}")

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(call_realtime_one, client, model, tok, prompt) for tok, prompt in requests_with_tokens]
            for fut in as_completed(futures):
                write_result(fut.result())
    finally:
        results_f.close()
        raw_f.close()

    return n_done - n_errored, n_errored, 0, in_tok, out_tok, cache_read, cache_create


def submit_batch(client, model, requests_with_tokens):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch_requests = []
    custom_id_to_token = {}

    for i, (token, prompt) in enumerate(requests_with_tokens):
        cid = f"tok_{i}"
        custom_id_to_token[cid] = token
        batch_requests.append(Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            ),
        ))

    batch = client.messages.batches.create(requests=batch_requests)
    return batch, custom_id_to_token


def save_batch_state(current_chunk_idx, total_chunks, current_batch_id, current_custom_id_to_token):
    BATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "current_chunk_idx": current_chunk_idx,
        "total_chunks": total_chunks,
        "current_batch_id": current_batch_id,
        "current_custom_id_to_token": current_custom_id_to_token,
        "submitted_at": datetime.now().isoformat(),
    }
    tmp = BATCH_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(BATCH_STATE_FILE)


def load_batch_state():
    if not BATCH_STATE_FILE.exists():
        return None
    try:
        with open(BATCH_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_batch_state():
    if BATCH_STATE_FILE.exists():
        BATCH_STATE_FILE.unlink()


def cancel_batch_safe(client, batch_id):
    try:
        client.messages.batches.cancel(batch_id)
        logging.info(f"  cancel sent for {batch_id}")
    except Exception as e:
        logging.warning(f"  cancel failed: {e}")


def poll_until_done(client, batch_id, timeout_sec=None):
    start = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        rc = batch.request_counts
        elapsed = int(time.time() - start)
        logging.info(
            f"  status={batch.processing_status} succeeded={rc.succeeded} "
            f"errored={rc.errored} processing={rc.processing} "
            f"canceled={getattr(rc, 'canceled', 0)} expired={getattr(rc, 'expired', 0)} "
            f"elapsed={elapsed}s"
        )
        if batch.processing_status == "ended":
            return batch, False
        if timeout_sec is not None and elapsed > timeout_sec and rc.succeeded == 0:
            logging.warning(f"  TIMEOUT: {elapsed}s with succeeded=0 (>{timeout_sec}s); canceling and skipping")
            cancel_batch_safe(client, batch_id)
            for _ in range(40):
                time.sleep(3)
                batch = client.messages.batches.retrieve(batch_id)
                if batch.processing_status == "ended":
                    break
            return batch, True
        time.sleep(POLL_INTERVAL_SEC)


def retrieve_and_write(client, batch_id, custom_id_to_token):
    HAIKU_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    n_succeeded = 0
    n_errored = 0
    n_unknown = 0
    in_tok = 0
    out_tok = 0
    cache_read = 0
    cache_create = 0

    results_f = open(HAIKU_RESULTS, "a", encoding="utf-8")
    raw_f = open(HAIKU_RAW_RESPONSES, "a", encoding="utf-8")
    try:
        for result in client.messages.batches.results(batch_id):
            cid = result.custom_id
            token = custom_id_to_token.get(cid)
            if token is None:
                continue
            if result.result.type == "succeeded":
                msg = result.result.message
                response_text = "".join(b.text for b in msg.content if b.type == "text")
                label = parse_label(response_text)
                if label == "UNKNOWN":
                    label = "NONE"
                    n_unknown += 1
                results_f.write(json.dumps({"token": token, "label": label}, ensure_ascii=False) + "\n")
                raw_f.write(json.dumps({"token": token, "response": response_text}, ensure_ascii=False) + "\n")
                in_tok += msg.usage.input_tokens
                out_tok += msg.usage.output_tokens
                cache_read += getattr(msg.usage, "cache_read_input_tokens", 0) or 0
                cache_create += getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
                n_succeeded += 1
            elif result.result.type == "errored":
                err = str(result.result.error) if hasattr(result.result, "error") else "unknown"
                raw_f.write(json.dumps({"token": token, "error": err}, ensure_ascii=False) + "\n")
                n_errored += 1
            else:
                raw_f.write(json.dumps({"token": token, "outcome": result.result.type}, ensure_ascii=False) + "\n")
                n_errored += 1
            results_f.flush()
            raw_f.flush()
    finally:
        results_f.close()
        raw_f.close()

    return n_succeeded, n_errored, n_unknown, in_tok, out_tok, cache_read, cache_create


def write_final_results(majority_rows, haiku_results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("token\tfinal_label\tsource\n")
        for row in majority_rows:
            tok = row["token"]
            if row["label"] == "NEOLOGISM" and tok in haiku_results:
                f.write(f"{tok}\t{haiku_results[tok]}\thaiku\n")
            else:
                f.write(f"{tok}\t{row['label']}\tmajority\n")


def is_complete():
    return COMPLETE_FLAG.exists()


def mark_complete():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETE_FLAG.touch()


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run(args):
    setup_logging()

    if is_complete() and not args.force:
        logging.info("Stage 9 already complete. Use --force to re-run.")
        return True

    if not MAJORITY_VOTE_RESULTS.exists():
        logging.error(f"Majority vote results not found: {MAJORITY_VOTE_RESULTS}")
        return False
    if not CANDIDATES_PRE_LLM.exists():
        logging.error(f"Candidates contexts not found: {CANDIDATES_PRE_LLM}")
        return False

    logging.info("=" * 70)
    logging.info("STAGE 9: HAIKU JUDGE (NEOLOGISM verifier on majority-vote candidates)")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model}")

    client = get_anthropic_client()

    if args.resubmit:
        clear_batch_state()

    state = load_batch_state()
    if state:
        logging.info(f"Found pending batch state: id={state['current_batch_id']} "
                     f"chunk={state['current_chunk_idx'] + 1}/{state['total_chunks']}")
        logging.info("Retrieving pending batch results before submitting new chunks...")
        batch = client.messages.batches.retrieve(state["current_batch_id"])
        if batch.processing_status != "ended":
            logging.info(f"Polling pending batch...")
            batch = poll_until_done(client, state["current_batch_id"])
        stats = retrieve_and_write(client, state["current_batch_id"], state["current_custom_id_to_token"])
        logging.info(f"  retrieved: succeeded={stats[0]} errored={stats[1]} parsed_as_NONE={stats[2]}")
        logging.info(f"  tokens: input={stats[3]:,} output={stats[4]:,} cache_read={stats[5]:,} cache_create={stats[6]:,}")
        clear_batch_state()

    logging.info(f"Loading majority vote results from {MAJORITY_VOTE_RESULTS}")
    majority_rows = load_majority_vote_results(MAJORITY_VOTE_RESULTS)
    logging.info(f"Total tokens in majority vote: {len(majority_rows):,}")

    filtered = [r for r in majority_rows if r.get("label") == "NEOLOGISM"]
    logging.info(f"Filter -> {len(filtered):,} NEOLOGISM-labeled candidates from majority vote")

    already_done = load_done_haiku_tokens(HAIKU_RESULTS)
    if already_done:
        logging.info(f"Resume: {len(already_done):,} tokens already classified by Haiku, skipping")

    to_classify = [r for r in filtered if r["token"] not in already_done]
    if args.max_tokens is not None:
        to_classify = to_classify[:args.max_tokens]
        logging.info(f"--max-tokens={args.max_tokens}: limiting this run to {len(to_classify):,} tokens")
    logging.info(f"Tokens to classify with Haiku this run: {len(to_classify):,}")

    if to_classify:
        target_tokens = {r["token"] for r in to_classify}
        logging.info(f"Loading contexts for {len(target_tokens):,} tokens...")
        contexts_by_token = load_candidate_contexts(CANDIDATES_PRE_LLM, target_tokens)
        missing = target_tokens - set(contexts_by_token.keys())
        if missing:
            logging.warning(f"  {len(missing)} tokens missing context; will classify with empty context")

        all_requests = []
        for r in to_classify:
            tok = r["token"]
            ctx = contexts_by_token.get(tok, [])
            prompt = build_user_prompt(tok, ctx)
            all_requests.append((tok, prompt))

        if args.realtime:
            logging.info(f"Running in REAL-TIME mode (no batch discount); concurrency={args.concurrency}")
            stats = run_realtime(client, args.model, all_requests, args.concurrency)
            logging.info(f"Realtime done: succeeded={stats[0]} errored={stats[1]}")
            logging.info(f"Tokens: input={stats[3]:,} output={stats[4]:,} cache_read={stats[5]:,} cache_create={stats[6]:,}")
        else:
            chunk_size = args.chunk_size
            timeout_sec = args.batch_timeout_min * 60
            chunks = list(chunked(all_requests, chunk_size))
            logging.info(f"Splitting into {len(chunks)} batch(es) of up to {chunk_size} (timeout {args.batch_timeout_min} min/batch)")

            n_skipped = 0
            for chunk_idx, chunk in enumerate(chunks):
                logging.info(f"--- Batch {chunk_idx + 1}/{len(chunks)} ({len(chunk):,} tokens) ---")
                logging.info(f"  Submitting...")
                batch, custom_id_to_token = submit_batch(client, args.model, chunk)
                save_batch_state(chunk_idx, len(chunks), batch.id, custom_id_to_token)
                logging.info(f"  Batch id={batch.id}; polling every {POLL_INTERVAL_SEC}s...")
                batch, timed_out = poll_until_done(client, batch.id, timeout_sec=timeout_sec)
                if timed_out:
                    logging.warning(f"  Chunk {chunk_idx + 1} TIMED OUT and was canceled; tokens will retry on next run")
                    n_skipped += 1
                logging.info(f"  Retrieving results...")
                stats = retrieve_and_write(client, batch.id, custom_id_to_token)
                logging.info(f"  Chunk done: succeeded={stats[0]} errored={stats[1]} parsed_as_NONE={stats[2]}")
                logging.info(f"  Tokens: input={stats[3]:,} output={stats[4]:,} cache_read={stats[5]:,} cache_create={stats[6]:,}")
                clear_batch_state()
            if n_skipped > 0:
                logging.warning(f"{n_skipped}/{len(chunks)} chunks timed out and were skipped. Re-run to retry.")

    logging.info(f"Building {HAIKU_JUDGE_RESULTS.name}...")
    haiku_results = load_done_haiku_tokens(HAIKU_RESULTS)
    write_final_results(majority_rows, haiku_results, HAIKU_JUDGE_RESULTS)
    logging.info(f"Wrote {len(majority_rows):,} rows to {HAIKU_JUDGE_RESULTS}")

    final_counts = Counter()
    source_counts = Counter()
    haiku_disagreement = 0
    for row in majority_rows:
        tok = row["token"]
        if row["label"] == "NEOLOGISM" and tok in haiku_results:
            final_counts[haiku_results[tok]] += 1
            source_counts["haiku"] += 1
            if haiku_results[tok] != row["label"]:
                haiku_disagreement += 1
        else:
            final_counts[row["label"]] += 1
            source_counts["majority"] += 1

    logging.info("=" * 70)
    logging.info("FINAL RESULTS")
    logging.info("=" * 70)
    logging.info("Label distribution:")
    for label, cnt in final_counts.most_common():
        logging.info(f"  {label}: {cnt:,}")
    logging.info("Source breakdown:")
    for src, cnt in source_counts.most_common():
        logging.info(f"  {src}: {cnt:,}")
    if source_counts.get("haiku", 0) > 0:
        pct = 100 * haiku_disagreement / source_counts["haiku"]
        logging.info(f"Haiku overruled the majority on {haiku_disagreement:,} / {source_counts['haiku']:,} verified tokens ({pct:.1f}%)")

    if not args.dry_run and args.max_tokens is None:
        mark_complete()
        logging.info(f"Marked complete: {COMPLETE_FLAG}")
    elif args.max_tokens is not None:
        logging.info(f"--max-tokens set: NOT marking complete; re-run without --max-tokens to finish.")

    logging.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stage 9: Claude Haiku as final verifier on the NEOLOGISM bucket of the majority vote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Design: Haiku acts as a final verifier on the NEOLOGISM-labeled subset of the
3-LLM majority vote. For every token where the majority vote says NEOLOGISM
and Haiku has rendered a verdict, haiku_4_5_judge_results.tsv uses Haiku's
label (which may downgrade the token to ENTITY / FOREIGN / NONE). For every
other token (label != NEOLOGISM, or NEOLOGISM-labeled but Haiku skipped it),
the majority vote stands.

Input:  data/output/majority_vote_results.tsv (stage 8 output)
        data/output/stage7_candidates_pre_llm.jsonl (token contexts)

Output: data/output/haiku_4_5_judge_results.tsv
  token | final_label | source
  source = 'haiku'    -> majority said NEOLOGISM and Haiku judged this token
  source = 'majority' -> all other rows (Haiku verdict ignored even if present)

The script reads majority_vote_results.tsv, keeps only NEOLOGISM-labeled rows
(both unanimous and 2-of-3 majority), and sends them to Claude Haiku 4.5 for
verification. Haiku's verdict wins for those rows; for every other token, the
majority vote stands.

Resume: tokens already present in results_haiku.jsonl are skipped automatically.

Examples:
  python stage_9_haiku_judge.py
  python stage_9_haiku_judge.py --force        # re-run even if complete flag exists
  python stage_9_haiku_judge.py --realtime     # use real-time API instead of Batch (2x cost)

Environment:
  ANTHROPIC_API_KEY must be set
""",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--resubmit", action="store_true",
                        help="Discard any existing batch state and start fresh")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if complete flag exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run end-to-end but do not set complete flag")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Limit this run to N tokens (results saved, complete flag NOT set; re-run without --max-tokens to finish)")
    parser.add_argument("--realtime", action="store_true",
                        help="Use real-time API instead of Batch API (2x cost, immediate response). Useful for tests when batch is slow.")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Thread pool size for --realtime mode (default: 5)")
    parser.add_argument("--chunk-size", type=int, default=BATCH_CHUNK_SIZE,
                        help=f"Requests per batch (default: {BATCH_CHUNK_SIZE}). Smaller = better for low-tier queue caps.")
    parser.add_argument("--batch-timeout-min", type=int, default=BATCH_TIMEOUT_SEC // 60,
                        help=f"Cancel and skip a batch if 0 succeeded after N minutes (default: {BATCH_TIMEOUT_SEC // 60})")
    args = parser.parse_args()
    success = run(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
