# Neologism Detection Pipeline

> 📄 **Reference paper:** Rossini, D. & van der Plas, L. *From 124 Million Tokens to 1,021 Neologisms: A Large-Scale Pipeline for Automatic Neologism Detection*. Accepted at the **NeoLLM 2026 workshop at LREC 2026**.
> Paper link: <!-- PAPER_URL --> *(to be added)*
>
> The paper provides the theoretical background and motivates several of the design choices in this pipeline (the reference-vocabulary filtering strategy, the multi-LLM ensemble + verifier architecture, the inflection-deduplication rules, and the choice of the four-label scheme). Users adapting this pipeline for their own work are strongly encouraged to read the paper first — many decisions that look arbitrary in the code are justified there.

A modular pipeline that mines newly-coined words and named entities from any large text corpus. It tokenizes, deduplicates, filters against a user-supplied reference vocabulary, runs language detection and LLM-based classification, and produces a labelled, optionally inflection-deduplicated final list of candidates.

The pipeline does not impose a single use case. The example walked through in this README is **English candidate-vocabulary mining from social-media comments with a 2015 reference cutoff**, but the same pipeline runs equally on any corpus, language, or temporal cutoff if you adjust the reference vocabularies and the LLM prompts accordingly (see *Adapting to other corpora and languages* below).

The default LLM prompts return one of four labels per token:

- **NEOLOGISM** — newly-coined words, slang, or derived forms (`doomscrolling`, `youtuber`, `instagrammable`)
- **ENTITY** — proper nouns (people, brands, products, fictional characters)
- **FOREIGN** — non-English words
- **NONE** — typos, junk, code identifiers, or anything else

These labels are defined inside the prompt files; you can rewrite them for any classification task.

## Use case

Useful when you have a corpus and want a curated list of new vocabulary candidates for lexicography, language-change research, terminology mining, or downstream NLP. The pipeline is designed to be reproducible at the scale of hundreds of millions of tokens, with each stage resumable from per-token JSONL outputs.

## Input format

The pipeline reads gzip-compressed CSV files from one or more directories under `NEOLOGISM_BASE_DIR`. The default subdirectory list is `processed_comments/`, which you can change in `config.py`:

```python
DATA_DIRS = [
    BASE_DIR / "your_corpus_subdir",
    # add more if you want to combine sources
]
```

Each CSV must contain **at least one of** the following text columns:
- `text`
- `body`

Optional columns:
- `id` — a unique identifier per text record. Used for cross-document deduplication in stage 2. If absent, deduplication is skipped.
- A topical/source field (e.g. `subreddit`, `domain`, `category`, …) — used to diversify context examples. The default is `subreddit`; you can rename it in `stage_6_build_context.py`.

## Pipeline stages

| Stage | Script | What it does | Cost profile |
|---|---|---|---|
| 0 | `stage_0_tokenization.py` | spaCy tokenization of every CSV. Lower-cases, strips URLs/emails, drops stop-words, writes tokenized output back to each CSV. | High CPU |
| 1 | `stage_1_merge_batches.py` | Merges per-file token counts into a single global counts file. | Mid CPU + RAM |
| 2 | `stage_2_duplicate_analysis.py` | Hashes record text to identify duplicates; produces `duplicate_ids.txt` so later stages can deduplicate when counting. | High CPU + RAM |
| 3 | `stage_3_token_counting.py` | Produces deduped global token counts using the dup-IDs from stage 2. | Mid CPU + RAM |
| 4 | `stage_4_vocab_filtering.py` | Removes tokens already in your reference vocabulary stack (e.g. an established lexicon for the language and time period you're studying). Uses SymSpell to also catch typos and concatenated words. | High CPU + RAM, longest stage |
| 5 | `stage_5_frequency_filtering.py` | Drops candidates below the `MIN_OCCURRENCES` threshold; runs token-level `lingua` language detection to filter clearly foreign words. **Prompts and language list are English-tuned by default** — see *Adapting* section. | Mid CPU |
| 6 | `stage_6_build_context.py` | For each remaining candidate, samples up to N occurrences from the original corpus to provide context for downstream classification. | High CPU + RAM |
| 7 | `stage_7_llm_classify.py` | Runs three independent open-weight LLMs (Qwen, Llama, Mistral by default) via vLLM. Each returns one of four labels per token, after a context-level second pass of `lingua`. **Prompts are English-tuned by default.** | GPU (4× ≥ 80 GB recommended) |
| 8 | `stage_8_majority_vote.py` | Consolidates labels from the three LLMs into a single per-token decision. Outputs `majority_vote_results.tsv`. **Optional**; can stop here if you want a single-vote ensemble. | Trivial |
| 9 | `stage_9_haiku_judge.py` | Optional 4th-opinion verifier on the NEOLOGISM bucket via Anthropic API (Claude Haiku 4.5 by default). Reads `majority_vote_results.tsv`, sends every NEOLOGISM-labeled row to Haiku, and writes `haiku_4_5_judge_results.tsv` where Haiku's verdict (which may downgrade a token to ENTITY / FOREIGN / NONE) replaces the majority label *for those rows only*. All non-NEOLOGISM rows keep their majority-vote label. **Optional**; skip entirely if you don't have an API key. | API quota |
| 10 | `stage_10_inflection_dedup.py` | Inflectional deduplication of the NEOLOGISM bucket: drops `-s`/`-es`/`-ies`/`-ing` variants when the base form is already present. **Rules are English-specific** — see *Adapting*. Auto-detects whether to read stage 9's output or stage 8's. | Trivial |

The pipeline is **resumable**: each stage writes a `.flag` file in `data/checkpoints/` on completion and skips work it has already done.

## How to run

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

(`en_core_web_lg` is ~750 MB. Smaller variants like `_md` or `_sm` will work if you edit the model name in `stage_0_tokenization.py`, with some loss of tokenization quality on noisy text.)

For stage 7, install vLLM in a separate venv (it pulls a pinned torch and is best kept isolated). vLLM has good wheel coverage on x86_64 + CUDA. On other architectures (e.g. aarch64), wheel availability varies — check vLLM's release page or build from source.

### 2. Build the filtering vocabularies

The pipeline filters candidate tokens against a stack of reference vocabularies stored in `vocabs/`. **The cutoff date and the choice of reference sources are entirely up to you** — the example below uses 2015-01-01 as the cutoff and assembles vocabularies from Wikipedia titles, Wiktionary, Urban Dictionary, and the corpus itself, but you can substitute anything that gives you a "known prior vocabulary" for your task.

**Auto-generated by scripts in this repo:**

```bash
python vocab_scripts/download_wikipedia_titles_pre2015.py --output vocabs/wikipedia_titles_pre2015_vocab.txt
python vocab_scripts/download_wiktionary_pre2015.py       --output vocabs/wiktionary_pre2015_vocab.txt
python vocab_scripts/urbanDict_download.py                --output-dir vocabs/

NEOLOGISM_PRE2015_INPUT_DIR=/path/to/processed_corpus \
NEOLOGISM_PRE2015_OUTPUT_DIR=vocabs/ \
python vocab_scripts/extract_pre2015_vocab.py \
  --cutoff-date 2015-01-01 \
  --output-name pre2015_vocab.txt

python build_frequency_dict.py
```

For a different cutoff (e.g. 2010 or 2020), just pass `--cutoff-date` to `extract_pre2015_vocab.py` — and adapt the Wikipedia/Wiktionary download scripts similarly if you want their snapshots aligned to the same date.

**Must be obtained externally** (each is its own license/distribution decision; the repo does not ship them):

| File | Where to get it |
|---|---|
| `wordnet_vocab.txt` | extract from NLTK's WordNet via `nltk.corpus.wordnet`, e.g. `python -c "import nltk; nltk.download('wordnet'); from nltk.corpus import wordnet; open('vocabs/wordnet_vocab.txt','w').writelines(w + '\n' for w in sorted(set(l.name() for s in wordnet.all_synsets() for l in s.lemmas())))"` |
| `noslang_word_list.txt` | a slang/jargon dictionary appropriate for your domain. The example shipped with this pipeline used a private list obtained from the lexicon's authors; you'll need to source your own (academic correspondence, open slang corpora, scraped term lists from a relevant lexicon, etc.). One token per line. |

You can also drop in *any* additional vocabulary files and add them to `VOCAB_FILES` in `config.py`. The vocabulary stage is just "is this token in any of these reference lists?" — what you feed it is your choice.

### 3. Configure environment variables

Copy `.env.example` to `.env` (or export directly in your shell). At minimum:

```bash
export NEOLOGISM_BASE_DIR=/path/to/data_root      # required for all stages
export ANTHROPIC_API_KEY=sk-ant-...               # only for stage 9 (optional)
export HF_HOME=/path/to/huggingface_cache         # only for stage 7 (model cache)
export HF_TOKEN=...                               # only for stage 7 (gated models)
```

### 4. Run the stages

Each stage is a standalone Python script. Run them in order, sequentially:

```bash
python stage_0_tokenization.py
python stage_1_merge_batches.py
python stage_2_duplicate_analysis.py
python stage_3_token_counting.py
python stage_4_vocab_filtering.py
python stage_5_frequency_filtering.py
python stage_6_build_context.py
python stage_7_llm_classify.py        # GPU recommended
python stage_8_majority_vote.py
python stage_9_haiku_judge.py                    # optional, requires ANTHROPIC_API_KEY
python stage_10_inflection_dedup.py
```

Each stage writes a `.flag` file in `data/checkpoints/` on completion and skips work it has already done. To force a re-run, pass `--force`. To run only one of the stage 7 LLMs:

```bash
python stage_7_llm_classify.py --model qwen_72b
```

If you skip stage 9, stage 10 will automatically read `majority_vote_results.tsv` (from stage 8) instead of `haiku_4_5_judge_results.tsv`.

For very large corpora, wrap any stage in your scheduler of choice (SLURM, PBS, k8s, etc.) — every script is self-contained and logs to stdout.

## Library requirements

Listed in `requirements.txt`:

| Package | Used by | Purpose |
|---|---|---|
| `pandas` | most stages | CSV / DataFrame handling |
| `spacy` (+ `en_core_web_lg`) | stage 0 | tokenization, stop-words |
| `tqdm` | all | progress bars |
| `symspellpy` | stage 4 | spelling correction + word segmentation |
| `lingua-language-detector` | stages 5, 7 | language detection |
| `nltk` | stages 0, 2 | sentence segmentation |
| `torch`, `transformers`, `accelerate` | stage 7 | model loading (replaced by vLLM in production) |
| `anthropic` | stage 9 | Claude API client (only if you run stage 9) |

For stage 7 you additionally need **vLLM** for high-throughput inference. Install in a dedicated venv on the GPU node:

```bash
pip install vllm
```

vLLM pulls its own pinned torch — keep it isolated from the CPU-stages venv.

## Compute requirements at scale

For a corpus of ~800 M tokens, ~4 M unique tokens, distilled to ~280 K candidates after vocab + frequency + foreign filtering (the example case shipped with this pipeline):

| Stage | Wall time | RAM | CPUs | GPUs |
|---|---|---|---|---|
| 0 | ~6 h | 100 GB | 16-128 | — |
| 1 | ~2 h | 200 GB | 16 | — |
| 2 | ~3 h | 200 GB | 16 | — |
| 3 | ~3 h | 200 GB | 16 | — |
| 4 | **~35 h** | 200 GB | 16 | — |
| 5 | ~4 h | 100 GB | 4 | — |
| 6 | ~6 h | 200 GB | 16 | — |
| 7 (all 3 LLMs, vLLM) | ~12-24 h | 200 GB host | 16 | 4× ≥ 80 GB |
| 8 | seconds | 8 GB | 1 | — |
| 9 (Haiku batch) | 1-12 h | 8 GB | 1 | — |
| 10 | seconds | 8 GB | 1 | — |

A workstation with ≥200 GB RAM and 16 cores is comfortable for stages 0-6 at this scale. Smaller corpora (under ~100 M tokens) run on a 64 GB workstation. Stage 7 is the only stage that genuinely requires multi-GPU; on smaller GPUs (e.g. 4× 40 GB), use `--quantize` or run only Qwen + Llama and skip Mistral.

## Adapting to other corpora and languages

The pipeline architecture generalizes, but **several places contain English-specific or use-case-specific defaults you must adapt**:

### 1. Stage 0 — spaCy model (`stage_0_tokenization.py`)
```python
nlp = spacy.load("en_core_web_lg")
```
Replace with the spaCy model for your target language (e.g. `xx_core_web_lg` where `xx` is the ISO code). Update the stop-word list import accordingly.

### 2. Stage 4 — reference vocabularies (`config.py:VOCAB_FILES`)
```python
VOCAB_FILES = [
    VOCAB_DIR / "wikipedia_titles_pre2015_vocab.txt",
    VOCAB_DIR / "wiktionary_pre2015_vocab.txt",
    VOCAB_DIR / "noslang_word_list.txt",
    VOCAB_DIR / "pre2015_vocab.txt",
    VOCAB_DIR / "urban_dict_pre2015_vocab.txt",
    VOCAB_DIR / "wordnet_vocab.txt",
]
```
Replace with whatever set of reference vocabulary files makes sense for your task. Format expected: **one token per line**, lower-cased, UTF-8. Add or remove files freely. The SymSpell frequency dictionary (`vocabs/symspell_frequency_dict.txt`) is read separately and must use the format `token<TAB>frequency` (one entry per line).

### 3. Stage 4 — rule-based filtering logic (English-specific)
The filtering logic in `stage_4_vocab_filtering.py` and `utils/filtering_utils.py` includes several **English-tuned components** beyond the vocabulary files:

- **SymSpell typo correction** uses the language-specific frequency dictionary. If a candidate is one or two edits away from a known word, it's dropped as a typo. For language X, replace `vocabs/symspell_frequency_dict.txt` with a target-language frequency dictionary in the same `token<TAB>frequency` format.
- **Word segmentation** uses the same SymSpell index to catch concatenated multi-word strings (`doomscroll` → `doom` + `scroll`) that were typed without spaces. Segmentation only finds valid splits in the language whose dictionary you supply.
- **Stop-word list** (`utils/filtering_utils.py`) is loaded from spaCy's English stop-words by default. For language X, swap the import for `spacy.lang.<X>.stop_words.STOP_WORDS`.
- **Lorem Ipsum filter** (also in `filtering_utils.py`) is a hard-coded list of *pseudo-Latin filler words* — Lorem Ipsum is not real Latin, it's the standard placeholder/dummy text used in design and publishing when the real content isn't ready yet. The same boilerplate string gets pasted into corpora in every language, so this filter is universally useful and should be kept as-is regardless of target language.
- **Token-validity regex** in `is_valid_candidate()` assumes ASCII / Latin script. For non-Latin-script languages (Cyrillic, CJK, Arabic, etc.) you may need to widen the allowed character classes.
- **Token length thresholds** in `config.py:VOCAB_FILTERING_CONFIG` (`min_token_length`, `max_token_length`, `min_word_length_typo`, `min_word_length_segmentation`) are tuned for English word distributions and may need adjustment for languages with longer average word lengths (Finnish, German compounds) or shorter ones (Chinese-romanized).

### 4. Stage 5 — language detection (`stage_5_frequency_filtering.py` + `language_constants.py`)
`RELEVANT_LANGUAGES` lists languages flagged *as foreign* (i.e. everything you want to discard). For target language *X*, change the set so *X* is the kept language and everything you want to filter out is in the list.

### 5. Stages 7, 8, 9 — LLM prompts (English-specific)
The prompts in:
- `stage_7_llm_classify.py` (`create_llm_prompt`, `create_single_token_prompt`)
- `stage_9_haiku_judge.py` (`SYSTEM_PROMPT`)

are written in English with English neologism/slang examples. **You must translate them** to your target language and update the few-shot examples. The four-label scheme (ENTITY / NEOLOGISM / FOREIGN / NONE) is also a choice — if your task needs different categories (e.g. *technical-term / general-vocabulary / brand / discard*), rewrite the prompts and update the parser in `parse_llm_response`.

Also note that `stage_8_majority_vote.py`'s vote-type labels (`unanimous`, `majority`, `tie`) are language-agnostic — no edits needed there.

### 6. Stage 10 — inflection rules (English-specific)
The suffix rules in `stage_10_inflection_dedup.py` (`-s`, `-es`, `-ies`, `-ing` drop; `-ed` keep) are **English-specific**. Other languages have entirely different inflection patterns:
- Spanish/Italian/Portuguese: gender (`-o/-a`), number (`-s/-es`), verb conjugation (`-ar/-er/-ir` and dozens of forms)
- German: case + gender + number, separable prefixes
- Slavic languages: rich case morphology
- Agglutinative languages (Finnish, Turkish, Korean): suffix stacking

You'll need to rewrite `candidate_bases()` for your language's morphology — or replace it with a proper lemmatizer for that language (spaCy's lemmatizer, Stanza, language-specific morphological analyzers).

### 7. Cutoff date (your choice)
The example pipeline uses 2015-01-01 as the "anything before this is established vocabulary" cutoff. Pass any other date to `extract_pre2015_vocab.py --cutoff-date YYYY-MM-DD`. The corresponding `pre2015_vocab.txt` filename is just a convention — you can rename freely (the reference is in `config.py`).

## Reproducibility notes

- LLM classifications use `temperature=0` for determinism; outputs may still differ slightly across runs due to non-deterministic CUDA kernels, but disagreements are below the level that matters for majority-voted labels.
- Stage 9 (Haiku) uses Anthropic's Batch API by default; results can vary slightly between API runs due to model updates on Anthropic's side. Pin `--model claude-haiku-4-5` (or whatever exact version you used) for reproducibility.
- Stages 1-6, 8, and 10 are fully deterministic given the input corpus.
- The reference vocabulary snapshot date is a methodological choice — adjust by regenerating `vocabs/` against a different date.

## Citation

If you use this pipeline in academic work, please cite the accompanying paper:

```bibtex
@inproceedings{rossini2026neologism,
  title     = {From 124 Million Tokens to 1,021 Neologisms: A Large-Scale Pipeline for Automatic Neologism Detection},
  author    = {Rossini, Diego and van der Plas, Lonneke},
  booktitle = {Proceedings of the NeoLLM Workshop at LREC 2026},
  year      = {2026},
  url       = {<!-- PAPER_URL -->}
}
```

## Disclaimer

This pipeline produces **filtered candidates**, not gold-standard labels. Manual verification is recommended before downstream use, especially for the **NEOLOGISM** bucket where polysemy, inflectional variation, and domain-specific jargon can introduce noise. The example use case shipped with the repo is academic research on English neologism detection in social-media discourse — the methodological choices behind that case are documented in the reference paper (see top of this README).
