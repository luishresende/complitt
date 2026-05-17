# CompLiTT — Evaluation / Replication Package

This folder reproduces the evaluation reported for the multi-task CompLiTT
pipeline. It scores the generated book summaries with three metrics:

| Metric | Script | Reference needed? | Cost |
|--------|--------|-------------------|------|
| **BERTScore** (XLM-RoBERTa) | `bertscore_rouge.py` | Human summary | Free (local, GPU optional) |
| **ROUGE-1/2/L** | `bertscore_rouge.py` | Human summary | Free (local) |
| **BooookScore** | `booookscore_eval.py` | None (coherence) | **Paid LLM API** |

Both scripts run directly from the results dataset (`complitt_results`) — no extra files
are required.

---

## 1. Get the data

Download the results dataset and unpack it anywhere on disk:

> **`complitt_results` dataset:** [Google Drive](https://drive.google.com/drive/folders/1lHQkm9mMvlwyQO3JoCed3J6uoq-lrmcb?usp=drive_link)

It follows the structure produced by the CompLiTT pipeline:

```
complitt_results/
├── English/
│   └── <book>/
│       └── providers/
│           ├── human/summary.txt              # human reference summary
│           ├── gemini-2.5-flash/summary.txt
│           ├── gpt-5-mini-2025-08-07/summary.txt
│           ├── grok-4-1-fast-reasoning/summary.txt
│           ├── meta-llama/Llama-3.1-8B-Instruct/summary.txt
│           ├── mistralai/Ministral-3-14B-Instruct-2512/summary.txt
│           └── Qwen/Qwen2.5-7B-Instruct/summary.txt
├── Spanish/ ...
└── Italian/ ...
```

The `human` provider holds the ground-truth summary; it is used as the
reference for BERTScore/ROUGE and is itself scored by BooookScore.

## 2. Install dependencies

For BERTScore + ROUGE:

```bash
pip install -r evaluation/requirements.txt
```

The first BERTScore run downloads the `xlm-roberta-large` model (~2 GB) from
Hugging Face. A CUDA GPU is used automatically if available.

For BooookScore, clone the modified fork and install its requirements (see
[§4](#4-booookscore)).

---

## 3. BERTScore + ROUGE

Compares every model's `summary.txt` against the matching `human` reference,
across all languages and books, and writes one CSV row per summary.

```bash
python evaluation/bertscore_rouge.py \
    --books-dir /path/to/complitt_results \
    --output evaluation/results/bertscore_rouge.csv
```

Output columns: `language, book, provider, bert_precision, bert_recall,
bert_f1, rouge1_p/r/f, rouge2_p/r/f, rougeL_p/r/f`.

## 4. BooookScore

BooookScore measures summary **coherence** with an LLM annotator — it needs no
reference summary.

It is **not vendored** in this repository. Clone the modified fork into
`evaluation/BooookScore/` (the path is git-ignored) and install its
dependencies:

```bash
git clone https://github.com/luishresende/BooookScore.git evaluation/BooookScore
pip install -r evaluation/BooookScore/requirements.txt
```

The fork carries a fix to the Anthropic client and reads API keys from the
environment; see its commit history for details.

Then run the evaluation in two phases:

```bash
# Phase A — offline & free: build the summ_<provider>.json files
python evaluation/booookscore_eval.py \
    --books-dir /path/to/complitt_results \
    --output-dir evaluation/results/booookscore \
    --summ-only

# Phase B — calls the evaluator API (paid)
python evaluation/booookscore_eval.py \
    --books-dir /path/to/complitt_results \
    --output-dir evaluation/results/booookscore \
    --api anthropic --model claude-sonnet-4-6
```

The experiments used **`claude-sonnet-4-6`** (Anthropic) as the evaluator.
`--api` also accepts `openai`, `gemini` and `xai`. API keys are read from
environment variables; a `.env` file in the repo root is loaded automatically:

```bash
export ANTHROPIC_API_KEY="your_key"   # for --api anthropic
```

Results are written to
`evaluation/results/booookscore/booookscore_results.csv` (`provider,
booookscore`). Annotations are cached in `annot_*.json`, so an interrupted run
resumes without re-spending on already-annotated books.

Use `--providers gemini-2.5-flash human ...` to score only a subset, and
`--booookscore-dir` if the fork was cloned elsewhere.
