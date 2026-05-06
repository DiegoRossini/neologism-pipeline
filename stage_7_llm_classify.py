#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from config import (
    CHECKPOINTS_DIR,
    LOG_DIR,
    SCRIPTS_DIR,
    OUTPUT_DIR,
    CANDIDATE_NEOLOGISMS_FILE,
    CONTEXT_INDEX_FILE,
    CLASSIFICATION_DIR,
    VOCAB_DIR,
)
from language_constants import RELEVANT_LANGUAGES

MAX_CONTEXTS_PER_TOKEN = 3
CONTEXT_MAX_WORDS = 110
CONTEXT_LANG_CONFIDENCE_THRESHOLD = 0.75
MIN_SUBREDDIT_EXAMPLES = 10

PREPARED_CONTEXTS_FILE = OUTPUT_DIR / "stage7_prepared_contexts.jsonl"
CANDIDATES_PRE_LLM_FILE = OUTPUT_DIR / "stage7_candidates_pre_llm.jsonl"
FOREIGN_CONTEXT_FILE = OUTPUT_DIR / "foreign_words_context_detected.txt"

PREP_CHUNK_SIZE = 2000
PREP_DEFAULT_WORKERS = max(1, min(16, os.cpu_count() or 1))

GPU_SHARD_SIZE = 2000
CHECKPOINT_INTERVAL_SEC = 60.0

VLLM_MAX_MODEL_LEN = 16384
VLLM_GPU_MEMORY_UTILIZATION = 0.92
VLLM_TENSOR_PARALLEL_SIZE = 4

MODEL_CONFIGS = [
    {
        "key": "qwen_72b",
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 256,
        "needs_hf_login": False,
    },
    {
        "key": "llama_70b",
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 256,
        "needs_hf_login": True,
    },
    {
        "key": "mistral_large",
        "name": "mistralai/Mistral-Large-Instruct-2411",
        "dtype": "bfloat16",
        "max_new_tokens": 256,
        "needs_hf_login": True,
    },
]

OVERALL_COMPLETE_FLAG = CHECKPOINTS_DIR / "stage7_llm_complete.flag"

def model_done_flag(model_key):
    return CHECKPOINTS_DIR / f"stage7_{model_key}_complete.flag"

def setup_logging():
    log_file = LOG_DIR / "stage_7_llm_ner.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_available_gpus():
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    except ImportError:
        return []

def load_llm_judge(model_config, device="auto", quantize=False):
    model_name = model_config["name"]
    dtype = model_config["dtype"]
    needs_hf_login = model_config["needs_hf_login"]

    if needs_hf_login:
        try:
            from huggingface_hub import login
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                login(token=hf_token)
                logging.info(f"[LLM Judge] Logged in to HuggingFace for {model_name}")
            else:
                logging.warning(f"[LLM Judge] No HF_TOKEN found; {model_name} may fail if gated")
        except Exception as e:
            logging.warning(f"[LLM Judge] HF login failed: {e}")

    from vllm import LLM

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        dtype=dtype,
        enable_prefix_caching=True,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
    )
    if quantize:
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
        logging.info("[LLM Judge] Using bitsandbytes 4-bit quantization (vLLM)")

    logging.info(f"[LLM Judge] Loading {model_name} with vLLM (TP={VLLM_TENSOR_PARALLEL_SIZE}, prefix_caching=ON, dtype={dtype})...")
    start_time = time.time()
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    elapsed = time.time() - start_time
    logging.info(f"[LLM Judge] Model loaded in {elapsed:.1f}s")
    return llm, tokenizer, False

def unload_model(llm, tokenizer, _unused):
    try:
        del llm
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    logging.info("[LLM Judge] Model unloaded, GPU memory freed")

def center_context_around_token(context, token, max_tokens=110):
    if not context or not token:
        return token

    words = context.split()

    if len(words) <= max_tokens:
        return context

    token_lower = token.lower()
    occurrences = []
    for i, word in enumerate(words):
        if token_lower in word.lower():
            occurrences.append(i)

    if not occurrences:
        return " ".join(words[:max_tokens])

    best_pos = occurrences[0]
    best_balance = min(best_pos, len(words) - best_pos - 1)

    for pos in occurrences[1:]:
        left_available = pos
        right_available = len(words) - pos - 1
        balance = min(left_available, right_available)
        if balance > best_balance:
            best_balance = balance
            best_pos = pos

    half_window = (max_tokens - 1) // 2

    left_available = best_pos
    right_available = len(words) - best_pos - 1

    if left_available < half_window:
        left_start = 0
        right_end = min(len(words), max_tokens)
    elif right_available < half_window:
        right_end = len(words)
        left_start = max(0, len(words) - max_tokens)
    else:
        left_start = best_pos - half_window
        right_end = best_pos + half_window + 1

    return " ".join(words[left_start:right_end])

def hash_context(context):
    if not context:
        return None
    return hashlib.md5(context.encode('utf-8')).hexdigest()

def select_diverse_contexts(contexts_with_metadata, token):
    if not contexts_with_metadata:
        return []

    filtered_contexts = []
    seen_hashes = set()

    for ctx in contexts_with_metadata:
        text = ctx.get('text', '')
        subreddit = ctx.get('subreddit', 'unknown')

        ctx_hash = hash_context(text)
        if ctx_hash and ctx_hash in seen_hashes:
            continue
        seen_hashes.add(ctx_hash)

        centered_text = center_context_around_token(text, token, max_tokens=CONTEXT_MAX_WORDS)

        filtered_contexts.append({
            'text': centered_text,
            'subreddit': subreddit,
            'original_length': len(text.split()) if text else 0,
        })

    if not filtered_contexts:
        return []

    by_subreddit = defaultdict(list)
    for ctx in filtered_contexts:
        by_subreddit[ctx['subreddit']].append(ctx)

    subreddit_counts = [(sub, len(ctxs)) for sub, ctxs in by_subreddit.items()]
    subreddit_counts.sort(key=lambda x: -x[1])

    qualifying_subreddits = []
    for i, (sub, count) in enumerate(subreddit_counts):
        if i == 0:
            qualifying_subreddits.append(sub)
        elif i < 3 and count >= MIN_SUBREDDIT_EXAMPLES:
            qualifying_subreddits.append(sub)
        elif i >= 3:
            break

    selected = []
    slots_per_subreddit = {}

    n = MAX_CONTEXTS_PER_TOKEN
    if len(qualifying_subreddits) == 1:
        slots_per_subreddit[qualifying_subreddits[0]] = n
    elif len(qualifying_subreddits) == 2:
        slots_per_subreddit[qualifying_subreddits[0]] = (n + 1) // 2
        slots_per_subreddit[qualifying_subreddits[1]] = n // 2
    else:
        per_sub = n // 3
        remainder = n % 3
        slots_per_subreddit[qualifying_subreddits[0]] = per_sub + (1 if remainder > 0 else 0)
        slots_per_subreddit[qualifying_subreddits[1]] = per_sub + (1 if remainder > 1 else 0)
        slots_per_subreddit[qualifying_subreddits[2]] = per_sub

    for sub in qualifying_subreddits:
        sub_contexts = by_subreddit[sub]
        sub_contexts.sort(key=lambda x: -x['original_length'])

        slots = slots_per_subreddit.get(sub, 0)
        for ctx in sub_contexts[:slots]:
            selected.append({
                'text': ctx['text'],
                'subreddit': sub,
            })

    if len(selected) < MAX_CONTEXTS_PER_TOKEN:
        already_selected_texts = {ctx['text'] for ctx in selected}

        for sub, count in subreddit_counts:
            if sub in qualifying_subreddits:
                continue
            for ctx in by_subreddit[sub]:
                if ctx['text'] not in already_selected_texts:
                    selected.append({
                        'text': ctx['text'],
                        'subreddit': sub,
                    })
                    already_selected_texts.add(ctx['text'])
                    if len(selected) >= MAX_CONTEXTS_PER_TOKEN:
                        break
            if len(selected) >= MAX_CONTEXTS_PER_TOKEN:
                break

    return selected

_GLOBAL_CTX_INDEX = None

def _prep_worker(token_chunk):
    out = []
    for token in token_chunk:
        raw = _GLOBAL_CTX_INDEX.get(token, []) if _GLOBAL_CTX_INDEX is not None else []
        ctxs = []
        for c in raw:
            if isinstance(c, dict):
                ctxs.append(c)
            else:
                ctxs.append({'text': c, 'subreddit': 'unknown'})
        selected = select_diverse_contexts(ctxs, token)
        out.append({"token": token, "contexts": selected})
    return out

def streaming_prepare_contexts(tokens, ctx_index, output_path, n_workers, chunk_size):
    already_prepared = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_prepared.add(rec['token'])
                except Exception:
                    pass
        logging.info(f"[Prep] {len(already_prepared):,} tokens already prepared (resume)")

    todo = [t for t in tokens if t not in already_prepared]
    if not todo:
        logging.info("[Prep] All tokens already prepared")
        return

    logging.info(f"[Prep] Preparing {len(todo):,} tokens with {n_workers} worker(s), chunk={chunk_size}")

    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]

    out_f = open(output_path, 'a', encoding='utf-8')

    if n_workers <= 1:
        global _GLOBAL_CTX_INDEX
        _GLOBAL_CTX_INDEX = ctx_index
        try:
            for chunk in tqdm(chunks, desc="Preparing contexts"):
                batch = _prep_worker(chunk)
                for rec in batch:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out_f.flush()
        finally:
            _GLOBAL_CTX_INDEX = None
            out_f.close()
        return

    _GLOBAL_CTX_INDEX = ctx_index
    try:
        gc.freeze()
    except Exception:
        pass

    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for batch in tqdm(pool.imap_unordered(_prep_worker, chunks), total=len(chunks), desc="Preparing contexts"):
                for rec in batch:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out_f.flush()
    finally:
        out_f.close()
        try:
            gc.unfreeze()
        except Exception:
            pass
        _GLOBAL_CTX_INDEX = None

def streaming_foreign_detection(prepared_path, candidates_path):
    if candidates_path.exists():
        logging.info(f"[Foreign] {candidates_path.name} already exists, skipping")
        return

    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        logging.warning("[Foreign] lingua not installed, copying prepared->candidates unfiltered")
        with open(prepared_path, 'r', encoding='utf-8') as fin, open(candidates_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                fout.write(line)
        return

    logging.info("[Foreign] Building language detector...")
    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()

    dataset_dict_path = VOCAB_DIR / "dataset_frequency_dict_deduped.txt"
    dataset_freq = {}
    if dataset_dict_path.exists():
        with open(dataset_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        dataset_freq[parts[0]] = int(parts[1])
                    except ValueError:
                        pass

    foreign_records = []
    n_in = n_kept = n_foreign = n_reintegrated = 0

    candidates_partial = candidates_path.with_suffix(candidates_path.suffix + ".partial")
    with open(prepared_path, 'r', encoding='utf-8') as fin, open(candidates_partial, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Foreign detection"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_in += 1
            token = rec['token']
            contexts = rec.get('contexts', [])
            combined_text = " ".join(ctx.get('text', '') for ctx in contexts)

            if not contexts or len(combined_text.strip()) < 20:
                fout.write(line if line.endswith('\n') else line + '\n')
                n_kept += 1
                continue

            lang = detector.detect_language_of(combined_text)

            if lang is None or lang == Language.ENGLISH or lang not in RELEVANT_LANGUAGES:
                fout.write(line if line.endswith('\n') else line + '\n')
                n_kept += 1
                continue

            conf = detector.compute_language_confidence(combined_text, lang)
            if conf < CONTEXT_LANG_CONFIDENCE_THRESHOLD:
                fout.write(line if line.endswith('\n') else line + '\n')
                n_kept += 1
                continue

            full_freq = dataset_freq.get(token, 0)
            if full_freq > 0:
                fout.write(line if line.endswith('\n') else line + '\n')
                n_kept += 1
                n_reintegrated += 1
            else:
                foreign_records.append((token, lang.name, conf))
                n_foreign += 1

    candidates_partial.rename(candidates_path)

    if foreign_records:
        with open(FOREIGN_CONTEXT_FILE, 'w', encoding='utf-8') as f:
            for token, lang, conf in sorted(foreign_records, key=lambda x: (-x[2], x[0])):
                f.write(f"{token}\t{lang}\t{conf:.3f}\n")
        logging.info(f"[Foreign] Report: {FOREIGN_CONTEXT_FILE}")

    logging.info(f"[Foreign] in={n_in:,} kept={n_kept:,} foreign={n_foreign:,} reintegrated={n_reintegrated:,}")

def create_llm_prompt(tokens_with_info):
    prompt = """TASK: Classify each token into ONE category.

ENTITY - Pure proper nouns only (real/fictional): people, characters, companies, brands, products, games, movies, places, apps
Examples: elon, pikachu, google, iphone, fortnite, reddit, tokyo

NEOLOGISM - New English words, slang, OR words derived from proper nouns
Examples: doomscrolling, ghosting, rizz, bussin, adulting, covidiot, youtuber, redditor, trumpian, instagrammable, uberize, googlable

FOREIGN - Non-English words
Examples: além, anspielung, yapmyorum, además

NONE - Usernames, typos, programming terms, unclear words

CRITICAL RULES:
1. Derived forms are NEOLOGISM (youtuber → NEOLOGISM, youtube → ENTITY)
2. When uncertain, classify as NONE
3. Use the context and subreddit to understand usage

TOKENS:

"""

    for item in tokens_with_info:
        token = item["token"]
        contexts = item.get("contexts", [])

        prompt += f"TOKEN: {token}\n"

        if contexts:
            for i, ctx in enumerate(contexts, 1):
                text = ctx.get('text', 'No context')
                subreddit = ctx.get('subreddit', 'unknown')
                prompt += f"  context_{i} (r/{subreddit}): \"{text}\"\n"
        else:
            prompt += "  context: No context available\n"

        prompt += "\n"

    prompt += """OUTPUT:
One classification per line as TOKEN:LABEL (ENTITY, NEOLOGISM, FOREIGN, or NONE).
No explanations.

Example:
google:ENTITY
googled:NEOLOGISM
spotify:ENTITY
doomscrolling:NEOLOGISM
além:FOREIGN
xyzabc123:NONE
"""

    return prompt

def create_single_token_prompt(item):
    token = item["token"]
    contexts = item.get("contexts", [])

    prompt = f"""Classify this token into ONE category: ENTITY, NEOLOGISM, FOREIGN, or NONE.

ENTITY - Pure proper nouns only (real/fictional): people, characters, companies, brands, products, games, movies, places, apps
NEOLOGISM - New English words, slang, OR words derived from proper nouns (youtuber, trumpian, instagrammable)
FOREIGN - Non-English words
NONE - Usernames, typos, programming terms, unclear words

TOKEN: {token}
"""

    if contexts:
        for i, ctx in enumerate(contexts, 1):
            text = ctx.get('text', 'No context')
            subreddit = ctx.get('subreddit', 'unknown')
            prompt += f"  context_{i} (r/{subreddit}): \"{text}\"\n"
    else:
        prompt += "  context: No context available\n"

    prompt += f"\nAnswer with ONLY the label: {token}:LABEL"

    return prompt

def run_llm_batch(llm, tokenizer, batch_of_groups, max_new_tokens=256, tokens_per_prompt=1):
    from vllm import SamplingParams

    effective_max_new_tokens = max(max_new_tokens, tokens_per_prompt * 8)
    max_input_tokens = VLLM_MAX_MODEL_LEN - effective_max_new_tokens

    valid_groups = []
    valid_prompts = []
    oversize_groups = []

    for group in batch_of_groups:
        if tokens_per_prompt > 1:
            prompt_text = create_llm_prompt(group)
        else:
            prompt_text = create_single_token_prompt(group[0])
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > max_input_tokens:
            oversize_groups.append(group)
            continue
        valid_groups.append(group)
        valid_prompts.append(text)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=effective_max_new_tokens,
    )

    responses = []
    if valid_prompts:
        outputs = llm.generate(valid_prompts, sampling_params, use_tqdm=False)
        by_index = {}
        for o in outputs:
            idx = getattr(o, "request_id", None)
            try:
                idx = int(idx) if idx is not None else None
            except (TypeError, ValueError):
                idx = None
            if idx is not None and 0 <= idx < len(valid_prompts):
                by_index[idx] = o.outputs[0].text
        if len(by_index) == len(valid_prompts):
            responses = [by_index[i] for i in range(len(valid_prompts))]
        else:
            responses = [o.outputs[0].text for o in outputs]

    return valid_groups, responses, oversize_groups

def parse_llm_response(response, expected_tokens):
    result = {}
    valid_labels = {"ENTITY", "NEOLOGISM", "FOREIGN", "NONE"}

    if isinstance(response, list):
        for token, resp in zip(expected_tokens, response):
            label = _extract_label(resp, token, valid_labels)
            result[token] = label
    else:
        expected_set = set(expected_tokens)
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                parts = line.rsplit(':', 1)
                if len(parts) == 2:
                    token = parts[0].strip()
                    label = parts[1].strip().upper()
                    if token in expected_set:
                        if label in valid_labels:
                            result[token] = label
                        elif "ENTITY" in label:
                            result[token] = "ENTITY"
                        elif "NEOLOGISM" in label:
                            result[token] = "NEOLOGISM"
                        elif "FOREIGN" in label:
                            result[token] = "FOREIGN"
                        else:
                            result[token] = "NONE"

    for token in expected_tokens:
        if token not in result:
            if len(expected_tokens) == 1:
                result[token] = _extract_label(response, token, valid_labels)
            else:
                result[token] = "UNKNOWN"

    return result

def _extract_label(response_text, token, valid_labels):
    text = response_text.strip().upper()

    for line in text.split('\n'):
        line = line.strip()
        if ':' in line:
            parts = line.rsplit(':', 1)
            label = parts[1].strip()
            if label in valid_labels:
                return label

    for label in valid_labels:
        if label in text:
            return label

    return "UNKNOWN"

def load_tokens(filepath):
    tokens = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    return tokens

def load_context_index(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_complete():
    return OVERALL_COMPLETE_FLAG.exists()

def maybe_mark_overall_complete():
    all_done = all(model_done_flag(m["key"]).exists() for m in MODEL_CONFIGS)
    if all_done:
        OVERALL_COMPLETE_FLAG.touch()
        logging.info(f"[Stage 7] All models complete, overall flag set: {OVERALL_COMPLETE_FLAG}")
        return True
    pending = [m["key"] for m in MODEL_CONFIGS if not model_done_flag(m["key"]).exists()]
    logging.info(f"[Stage 7] Not marking overall complete; still pending: {pending}")
    return False

def _load_done_results(results_path):
    done = {}
    if not results_path.exists():
        return done
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            label = rec.get('label')
            tok = rec.get('token')
            if tok and label and label not in ("ERROR", "UNKNOWN"):
                done[tok] = label
    return done

def _shard_iter(candidates_path, shard_size, skip_tokens):
    shard = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('token') in skip_tokens:
                continue
            shard.append(rec)
            if len(shard) >= shard_size:
                yield shard
                shard = []
    if shard:
        yield shard

def run_single_model_streaming(model_config, candidates_path, timestamp, quantize=False, tokens_per_prompt=10):
    model_key = model_config["key"]
    model_name = model_config["name"]
    max_new_tokens = model_config["max_new_tokens"]

    flag = model_done_flag(model_key)
    if flag.exists():
        logging.info(f"[{model_key}] Already complete (flag exists), skipping")
        return

    logging.info("=" * 70)
    logging.info(f"MODEL: {model_key} ({model_name})")
    logging.info("=" * 70)

    results_path = CLASSIFICATION_DIR / f"results_{model_key}.jsonl"
    raw_responses_path = CLASSIFICATION_DIR / f"raw_responses_{model_key}.jsonl"

    done_results = _load_done_results(results_path)
    skip_set = set(done_results.keys())
    logging.info(f"[{model_key}] {len(skip_set):,} tokens already classified successfully (will skip)")

    n_remaining = 0
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('token') not in skip_set:
                n_remaining += 1
    logging.info(f"[{model_key}] {n_remaining:,} tokens remaining | tokens_per_prompt={tokens_per_prompt} | shard_size={GPU_SHARD_SIZE}")

    if n_remaining == 0:
        flag.touch()
        logging.info(f"[{model_key}] Nothing to do; marked complete")
        return

    llm, tokenizer, _ = load_llm_judge(model_config, quantize=quantize)

    results_f = open(results_path, 'a', encoding='utf-8')
    raw_f = open(raw_responses_path, 'a', encoding='utf-8')

    def flush_files():
        try:
            results_f.flush()
            raw_f.flush()
            os.fsync(results_f.fileno())
            os.fsync(raw_f.fileno())
        except Exception as e:
            logging.warning(f"[{model_key}] flush failed: {e}")

    def sigterm_handler(signum, frame):
        logging.warning(f"[{model_key}] SIGTERM received, flushing and exiting")
        flush_files()
        try:
            results_f.close()
            raw_f.close()
        except Exception:
            pass
        sys.exit(0)

    prev_handler = signal.signal(signal.SIGTERM, sigterm_handler)

    pending_retry = []
    last_flush_t = time.time()
    n_done_this_run = 0
    start_time = time.time()

    try:
        for shard_idx, shard in enumerate(_shard_iter(candidates_path, GPU_SHARD_SIZE, skip_set)):
            shard_t0 = time.time()
            groups = [shard[i:i + tokens_per_prompt] for i in range(0, len(shard), tokens_per_prompt)]
            logging.info(f"[{model_key}] Shard {shard_idx} | {len(shard)} tokens | {len(groups)} prompts")

            try:
                valid_groups, responses, oversize_groups = run_llm_batch(
                    llm, tokenizer, groups,
                    max_new_tokens, tokens_per_prompt
                )
                if oversize_groups:
                    logging.warning(f"[{model_key}] shard{shard_idx}: {len(oversize_groups)} prompt(s) over max_input_tokens, routing to single-token retry")
                    for g in oversize_groups:
                        for it in g:
                            pending_retry.append(it)

                for g, resp in zip(valid_groups, responses):
                    gtoks = [it["token"] for it in g]
                    raw_f.write(json.dumps({"shard": shard_idx, "tokens": gtoks, "response": resp}, ensure_ascii=False) + '\n')

                shard_results = {}
                for g, resp in zip(valid_groups, responses):
                    gtoks = [it["token"] for it in g]
                    shard_results.update(parse_llm_response(resp, gtoks))

                for token, label in shard_results.items():
                    if label in ("ERROR", "UNKNOWN"):
                        t_item = next((it for g in valid_groups for it in g if it["token"] == token), None)
                        if t_item is not None:
                            pending_retry.append(t_item)
                    else:
                        results_f.write(json.dumps({"token": token, "label": label}, ensure_ascii=False) + '\n')
                        n_done_this_run += 1
            except Exception as e:
                logging.error(f"[{model_key}] Error shard{shard_idx}: {e}")
                for g in groups:
                    for it in g:
                        pending_retry.append(it)

            flush_files()
            shard_dt = time.time() - shard_t0
            elapsed = time.time() - start_time
            cumul_rate = n_done_this_run / elapsed if elapsed > 0 else 0
            shard_rate = len(shard) / shard_dt if shard_dt > 0 else 0
            logging.info(f"[{model_key}] shard{shard_idx} done in {shard_dt:.1f}s | shard_rate={shard_rate:.1f} tok/s | cumul_rate={cumul_rate:.1f} tok/s | done_this_run={n_done_this_run:,} | pending_retry={len(pending_retry):,}")
            last_flush_t = time.time()

        if pending_retry:
            logging.info(f"[{model_key}] Retrying {len(pending_retry):,} ERROR/UNKNOWN tokens (single-token prompts)")
            retry_groups = [[item] for item in pending_retry]

            try:
                valid_retry_groups, responses, oversize_retry_groups = run_llm_batch(
                    llm, tokenizer, retry_groups,
                    max_new_tokens, tokens_per_prompt=1
                )
                if oversize_retry_groups:
                    logging.warning(f"[{model_key}] retry: {len(oversize_retry_groups)} single-token prompt(s) STILL over limit, defaulting to NONE")
                    for g in oversize_retry_groups:
                        results_f.write(json.dumps({"token": g[0]["token"], "label": "NONE"}, ensure_ascii=False) + '\n')
                        n_done_this_run += 1

                for g, resp in zip(valid_retry_groups, responses):
                    raw_f.write(json.dumps({"retry": True, "tokens": [g[0]["token"]], "response": resp}, ensure_ascii=False) + '\n')

                retry_results = {}
                for g, resp in zip(valid_retry_groups, responses):
                    gtoks = [g[0]["token"]]
                    retry_results.update(parse_llm_response(resp, gtoks))

                for token, label in retry_results.items():
                    if label in ("ERROR", "UNKNOWN"):
                        label = "NONE"
                    results_f.write(json.dumps({"token": token, "label": label}, ensure_ascii=False) + '\n')
                    n_done_this_run += 1
            except Exception as e:
                logging.error(f"[{model_key}] Retry error: {e}")
                for g in retry_groups:
                    results_f.write(json.dumps({"token": g[0]["token"], "label": "NONE"}, ensure_ascii=False) + '\n')
                    n_done_this_run += 1

            flush_files()
    finally:
        flush_files()
        try:
            results_f.close()
        except Exception:
            pass
        try:
            raw_f.close()
        except Exception:
            pass
        signal.signal(signal.SIGTERM, prev_handler)
        unload_model(llm, tokenizer, None)

    elapsed = time.time() - start_time
    logging.info(f"[{model_key}] Pass complete in {elapsed:.1f}s | classified_this_run={n_done_this_run:,}")

    final_results = _load_done_results(results_path)
    final_stats = {
        "entity": sum(1 for v in final_results.values() if v == "ENTITY"),
        "neologism": sum(1 for v in final_results.values() if v == "NEOLOGISM"),
        "foreign": sum(1 for v in final_results.values() if v == "FOREIGN"),
        "none": sum(1 for v in final_results.values() if v == "NONE"),
    }
    logging.info(f"[{model_key}] Final: {final_stats}")

    summary_path = CLASSIFICATION_DIR / f"summary_{model_key}_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "model_key": model_key,
        "model_name": model_name,
        "total_classified": len(final_results),
        "stats": final_stats,
        "results_jsonl": str(results_path),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info(f"[{model_key}] Summary saved to {summary_path}")

    tsv_path = CLASSIFICATION_DIR / f"classified_{model_key}_{timestamp}.tsv"
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("token\tlabel\tis_entity\n")
        for token in sorted(final_results.keys()):
            label = final_results[token]
            is_entity = "true" if label == "ENTITY" else "false"
            f.write(f"{token}\t{label}\t{is_entity}\n")
    logging.info(f"[{model_key}] TSV saved to {tsv_path}")

    flag.touch()
    logging.info(f"[{model_key}] Marked complete: {flag}")

def generate_slurm_script(
    job_name="neologism_ner",
    time_limit="04:00:00",
    num_gpus=4,
    partition="normal",
    account=None,
):
    account_line = f"#SBATCH --account={account}" if account else ""

    script = f'''#!/bin/bash
{account_line}

cd {SCRIPTS_DIR}
python stage_7_llm_classify.py

echo "Job completed at $(date)"
'''
    return script

def submit_slurm_job(account=None, time_limit="04:00:00", num_gpus=4, partition="normal"):
    import subprocess

    script_content = generate_slurm_script(
        account=account,
        time_limit=time_limit,
        num_gpus=num_gpus,
        partition=partition,
    )

    slurm_script_path = SCRIPTS_DIR / "run_stage7.slurm"
    with open(slurm_script_path, 'w') as f:
        f.write(script_content)

    logging.info(f"SLURM script written to {slurm_script_path}")

    try:
        result = subprocess.run(
            ["sbatch", str(slurm_script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"Job submitted successfully. Job ID: {job_id}")
            logging.info(f"Monitor with: squeue -j {job_id}")
            logging.info(f"Logs will be at: {LOG_DIR}/slurm_{job_id}.out")
            return job_id
        else:
            logging.error(f"Failed to submit job: {result.stderr}")
            return None

    except FileNotFoundError:
        logging.error("sbatch command not found. Are you on a SLURM cluster?")
        return None

def _phase3_subprocess_target(model_config, candidates_path_str, timestamp, quantize, tokens_per_prompt):
    from pathlib import Path
    setup_logging()
    run_single_model_streaming(
        model_config, Path(candidates_path_str), timestamp,
        quantize=quantize, tokens_per_prompt=tokens_per_prompt,
    )


def run(
    force=False,
    llm_device="auto",
    dry_run=False,
    model_filter=None,
    quantize=False,
    tokens_per_prompt=10,
    max_tokens=None,
    prep_workers=PREP_DEFAULT_WORKERS,
    skip_prep=False,
    skip_foreign=False,
):
    setup_logging()

    if is_complete() and not force:
        logging.info("Stage 7 already complete. Skipping. Use force=True to re-run.")
        return True

    logging.info("=" * 70)
    logging.info("STAGE 7: LLM NER FILTERING (Multi-Model Independent)")
    logging.info("=" * 70)

    if not CANDIDATE_NEOLOGISMS_FILE.exists():
        logging.error(f"Prerequisite not found: {CANDIDATE_NEOLOGISMS_FILE}")
        return False

    if not CONTEXT_INDEX_FILE.exists():
        logging.error(f"Prerequisite not found: {CONTEXT_INDEX_FILE}")
        return False

    if model_filter:
        models_to_run = [m for m in MODEL_CONFIGS if m["key"] == model_filter]
        if not models_to_run:
            logging.error(f"Unknown model key: {model_filter}. Available: {[m['key'] for m in MODEL_CONFIGS]}")
            return False
        logging.info(f"Running single model: {model_filter}")
    else:
        models_to_run = MODEL_CONFIGS
        logging.info(f"Running all {len(models_to_run)} models sequentially")

    for mc in models_to_run:
        logging.info(f"  - {mc['key']}: {mc['name']}")

    logging.info("Loading tokens...")
    tokens = load_tokens(CANDIDATE_NEOLOGISMS_FILE)

    if dry_run:
        tokens = tokens[:100]
        logging.info(f"DRY RUN: Processing only first 100 tokens")

    if max_tokens is not None:
        tokens = tokens[:max_tokens]
        logging.info(f"--max-tokens: Limited to first {max_tokens:,} tokens ({len(tokens):,} actual)")

    logging.info(f"Loaded {len(tokens):,} tokens")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if skip_prep:
        logging.info("[Prep] Skipped (--skip-prep)")
        if not PREPARED_CONTEXTS_FILE.exists():
            logging.error(f"--skip-prep requested but {PREPARED_CONTEXTS_FILE} does not exist")
            return False
    else:
        prep_complete = PREPARED_CONTEXTS_FILE.exists() and _count_jsonl_lines(PREPARED_CONTEXTS_FILE) >= len(tokens)
        if prep_complete:
            logging.info(f"[Prep] {PREPARED_CONTEXTS_FILE.name} already covers all {len(tokens):,} tokens, skipping prep")
        else:
            logging.info("=== Phase 1: Streaming context preparation ===")
            ctx_index = load_context_index(CONTEXT_INDEX_FILE)
            logging.info(f"[Prep] Loaded context_index ({len(ctx_index):,} keys)")
            streaming_prepare_contexts(tokens, ctx_index, PREPARED_CONTEXTS_FILE,
                                       n_workers=prep_workers, chunk_size=PREP_CHUNK_SIZE)
            del ctx_index
            gc.collect()

    if not skip_foreign:
        logging.info("=== Phase 2: Streaming foreign language detection ===")
        streaming_foreign_detection(PREPARED_CONTEXTS_FILE, CANDIDATES_PRE_LLM_FILE)
    else:
        logging.info("[Foreign] Skipped (--skip-foreign)")
        if not CANDIDATES_PRE_LLM_FILE.exists():
            import shutil
            shutil.copy(PREPARED_CONTEXTS_FILE, CANDIDATES_PRE_LLM_FILE)

    needs_gpu = any(not model_done_flag(m["key"]).exists() for m in models_to_run)
    if needs_gpu:
        gpu_available = check_gpu_available()
        available_gpus = get_available_gpus()
        logging.info(f"GPU available: {gpu_available}")
        if gpu_available:
            logging.info(f"Available GPUs: {available_gpus}")
        else:
            logging.error("No GPU available. Phases 1-2 finished; phase 3 requires GPU. Exiting.")
            return False

    logging.info("=== Phase 3: Per-model GPU classification (vLLM) ===")

    pending_models = [mc for mc in models_to_run if not model_done_flag(mc["key"]).exists()]
    if len(pending_models) <= 1:
        for model_config in models_to_run:
            run_single_model_streaming(
                model_config, CANDIDATES_PRE_LLM_FILE, timestamp,
                quantize=quantize, tokens_per_prompt=tokens_per_prompt
            )
    else:
        logging.info(f"[Stage 7] {len(pending_models)} models to run; dispatching each in a fresh subprocess for clean GPU state")
        ctx = mp.get_context("spawn")
        for mc in models_to_run:
            if model_done_flag(mc["key"]).exists():
                logging.info(f"[Stage 7] {mc['key']} already complete, skipping")
                continue
            logging.info(f"[Stage 7] Spawning subprocess for {mc['key']}")
            p = ctx.Process(
                target=_phase3_subprocess_target,
                args=(mc, str(CANDIDATES_PRE_LLM_FILE), timestamp, quantize, tokens_per_prompt),
            )
            p.start()
            p.join()
            if p.exitcode != 0:
                logging.error(f"[Stage 7] Subprocess for {mc['key']} exited with code {p.exitcode}; continuing with next model")

    if not dry_run and max_tokens is None:
        maybe_mark_overall_complete()

    logging.info("=" * 70)
    logging.info("STAGE 7 PHASE FINISHED")
    logging.info(f"Models run: {[m['key'] for m in models_to_run]}")
    logging.info(f"Total tokens loaded: {len(tokens):,}")
    logging.info(f"Per-model outputs in: {CLASSIFICATION_DIR}")
    logging.info("=" * 70)

    return True

def _count_jsonl_lines(path):
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            n += 1
    return n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 7: LLM NER Filtering (Multi-Model Independent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage_7_llm_classify.py
  python stage_7_llm_classify.py --model qwen_72b
  python stage_7_llm_classify.py --model llama_70b
  python stage_7_llm_classify.py --model mistral_large
  python stage_7_llm_classify.py --dry-run
  python stage_7_llm_classify.py --force

Models:
  qwen_72b      - Qwen/Qwen2.5-72B-Instruct (standard)
  llama_70b     - meta-llama/Llama-3.3-70B-Instruct (needs HF_TOKEN)
  mistral_large - mistralai/Mistral-Large-Instruct-2411 (dense, 123B)

Classification:
  ENTITY    - Pure proper nouns
  NEOLOGISM - New words / slang / entity derivatives
  FOREIGN   - Non-English words
  NONE      - Unclear / manual review
        """
    )

    parser.add_argument("--model", type=str, default=None,
                        choices=["qwen_72b", "llama_70b", "mistral_large"],
                        help="Run only this model (default: all 3 sequentially)")

    parser.add_argument("--llm-device", type=str, default="auto",
                        help="Device for LLM (default: auto)")

    parser.add_argument("--submit-slurm", action="store_true",
                        help="Submit job to SLURM instead of running locally")
    parser.add_argument("--account", type=str, default=None,
                        help="SLURM account")
    parser.add_argument("--time-limit", type=str, default="24:00:00",
                        help="SLURM time limit (default: 24:00:00)")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs for SLURM job (default: 4)")
    parser.add_argument("--partition", type=str, default="normal",
                        help="SLURM partition (default: normal)")

    parser.add_argument("--tokens-per-prompt", type=int, default=10,
                        help="Tokens per multi-token prompt (default: 10)")

    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit NF4 quantization (bitsandbytes)")

    parser.add_argument("--prep-workers", type=int, default=PREP_DEFAULT_WORKERS,
                        help=f"Workers for context prep (default: {PREP_DEFAULT_WORKERS}). 1 disables multiprocessing.")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip context preparation phase (assumes JSONL exists)")
    parser.add_argument("--skip-foreign", action="store_true",
                        help="Skip foreign language detection")

    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if complete")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only first 100 tokens for testing")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Limit to first N tokens (e.g. --max-tokens 35000)")

    args = parser.parse_args()

    if args.submit_slurm:
        setup_logging()
        job_id = submit_slurm_job(
            account=args.account,
            time_limit=args.time_limit,
            num_gpus=args.num_gpus,
            partition=args.partition,
        )
        sys.exit(0 if job_id else 1)

    success = run(
        force=args.force,
        llm_device=args.llm_device,
        dry_run=args.dry_run,
        model_filter=args.model,
        quantize=args.quantize,
        tokens_per_prompt=args.tokens_per_prompt,
        max_tokens=args.max_tokens,
        prep_workers=args.prep_workers,
        skip_prep=args.skip_prep,
        skip_foreign=args.skip_foreign,
    )

    sys.exit(0 if success else 1)
