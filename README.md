# CompLiTT — A Multi-Task LLM Pipeline for Historical Books

CompLiTT is a command-line pipeline that processes historical books in PDF format and extracts structured knowledge from them using modern OCR and Large Language Models (LLMs).

This repository contains the **extended, multi-task version** of CompLiTT. Compared to the original pipeline, this version introduces three contributions:

* **Adaptive summarisation** — the summarisation step iteratively compresses a book until it fits the model's context window, instead of relying on fixed-size blocks.
* **Multiple downstream tasks** — from the same compressed content, the pipeline produces a **summary**, **keywords**, and **thematic categories**.
* **Open-source + proprietary models** — nine providers are supported, covering both API-based proprietary models and locally executed open-source models.

You can check our results [here](https://drive.google.com/drive/folders/1mMKY8ry4GjUa8ofzVkv5VJKde4sF3D_A?usp=drive_link).

---

## 📚 Publications & Versions

| Version | Paper | Code |
|---------|-------|------|
| **v2 — Multi-task adaptive pipeline** *(this branch)* | Paper under review — details will be added once published | current branch |
| **v1 — Original CompLiTT** | *"CompLiTT: An end-to-end pipeline for processing and summarizing historical books"*, ICOLAA 2026 — accepted, in press | [`v1` branch](https://github.com/luishresende/complitt/tree/v1) |

> The original pipeline that accompanies the first paper is preserved in the [`v1` branch](https://github.com/luishresende/complitt/tree/v1).

---

## 📦 Overview

CompLiTT converts collections of historical PDFs into structured knowledge through the following pipeline:

```
PDFs → Organized Dataset → OCR → Clean Text → Adaptive Summarisation → Summary / Keywords / Categories
```

The flow has two phases:

1. **OCR phase** (no LLM) — extract and clean text from the scanned PDFs.
2. **LLM phase** — adaptively compress each book into intermediate abstracts (`resume`), then derive the downstream outputs (`abstract`, `keywords`, `categories`) from those abstracts.

Each step is exposed as a CLI command.

---

## 📁 Expected Directory Structure

Books must be organized by **language**, with one folder per language.

⚠️ The folder name is **not arbitrary**: it is used as the target language for the
generated outputs. Each book is summarised, and its keywords and categories are produced,
in the language given by its parent folder. Name the folders with the actual language
(e.g. `English`, `Spanish`, `Italian`).

```
input_dir/
├── English/
│   ├── book_a.pdf
│   └── book_b.pdf
├── Spanish/
│   └── book_c.pdf
```

CompLiTT enforces this hierarchy:

```
language → book → pdf
```

The `prepare_dir` command turns this layout into the structure the pipeline expects, creating one folder per book. As the pipeline runs, each book folder is populated like this:

```
language/book/
├── book.pdf
├── paddle_output/              # raw PaddleOCR output (.json + .md)
├── book_content.md             # post-processed text (optional / auto-generated)
└── providers/
    └── <model-name>/
        ├── meta.json           # batch / split bookkeeping
        ├── batches/            # intermediate summarisation rounds
        ├── abstract_contents.txt        # compressed book content
        ├── summary.txt                  # final summary
        ├── keywords.txt
        └── categories.txt
```

Outputs are stored **per model**, so multiple models can be run over the same dataset without overwriting each other's results.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/luishresende/complitt.git
cd complitt
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

* Python ≥ 3.10
* API keys for the proprietary providers you intend to use
* A CUDA-capable GPU is **required** to run the open-source models (`ministral`, `llama`, `qwen`) locally, and optional to accelerate OCR

### Install PaddleOCR

CompLiTT relies on **PaddleOCR** for text extraction. You must install PaddlePaddle and PaddleOCR following the official instructions, since installation depends on your system configuration (CPU / GPU / CUDA version).

👉 Official guide: https://github.com/PaddlePaddle/PaddleOCR

---

## 🚀 Usage

All commands are executed through:

```bash
python run.py <command> [arguments]
```

Run `python run.py <command> --help` for the full list of options of any command.

### 1️⃣ Prepare Directory

Organizes PDFs into the required structure by creating one folder per book and moving each PDF into its corresponding directory.

```bash
python run.py prepare_dir <input_dir>
```

It traverses language folders, creates a folder for each PDF, moves the PDF into it, and skips books that are already organized.

### 2️⃣ Run OCR (PaddleOCR)

Extracts text and layout information from all books using PaddleOCR. GPU is used automatically if available.

```bash
python run.py run_paddle <input_dir> [--output_relative_dir paddle_output]
```

For each book, results are written to `book/paddle_output/` (`.json` + `.md`).

### 3️⃣ Post-Process OCR Output (optional)

Cleans and merges OCR results into a single Markdown file per book — merging fragmented blocks across pages, recomposing hyphenated words, and normalising tables.

```bash
python run.py post_process_paddle <input_dir> \
    [--paddle_output_relative_dir paddle_output] \
    [--save_output_relative_path book_content.md]
```

⚠️ This step is **optional**: it is executed automatically during the `resume` step.

### 4️⃣ Resume — Adaptive Summarisation

Compresses each book into intermediate abstracts. The book text is split into chunks bounded by the context window, each chunk is summarised to roughly 10% of its size, and the process repeats iteratively until the concatenated result fits within the context window. The result (`abstract_contents.txt`) is the **unified input for all downstream tasks** and must be generated before `abstract`, `keywords`, and `categories`.

```bash
python run.py resume <input_dir> <api> <model> <context_window_size> \
    [--num_threads 1] \
    [--paddle_output_relative_dir paddle_output] \
    [--resume_version N] \
    [--max_retries -1]
```

* `context_window_size` — context window size **in characters** of the model in use.
* `--num_threads` — parallelism for processing the splits of a batch (default: `1`).

Example:

```bash
python run.py resume ./books google gemini-2.5-flash 32000
```

### 5️⃣ Downstream Tasks

Each downstream command loads the abstracts produced by `resume` and generates its output. They are independent and can be run in any order once `resume` has completed.

```bash
# Final book summary (summary.txt + summary_without_intro.txt)
python run.py abstract <input_dir> <api> <model> [--abstract_version N] [--max_retries -1]

# Keyword / theme extraction (keywords.txt)
python run.py keywords <input_dir> <api> <model> [--keywords_version N] [--max_retries -1]

# Thematic categorisation into knowledge areas (categories.txt)
python run.py categories <input_dir> <api> <model> [--categories_version N] [--max_retries -1]
```

All commands are **idempotent**: outputs that already exist are skipped, so interrupted runs can be safely resumed.

---

## 🤖 Supported Models

The `<api>` argument selects the provider; `<model>` is the provider-specific model identifier.

| `api`       | Type        | Provider   | Execution            |
|-------------|-------------|------------|----------------------|
| `google`    | Proprietary | Google     | API                  |
| `anthropic` | Proprietary | Anthropic  | API                  |
| `openai`    | Proprietary | OpenAI     | API                  |
| `xai`       | Proprietary | xAI        | API                  |
| `mistral`   | Proprietary | Mistral AI | API                  |
| `deepseek`  | Proprietary | DeepSeek   | API                  |
| `ministral` | Open-source | Mistral AI | Local (Hugging Face) |
| `llama`     | Open-source | Meta       | Local (Hugging Face) |
| `qwen`      | Open-source | Alibaba    | Local (Hugging Face) |

Open-source models are downloaded from Hugging Face and run locally on the GPU. The proprietary providers support `--max_retries` for transient API errors (rate limits, server errors); `-1` means infinite retries.

---

## 🔑 API Configuration

Set the API keys for the proprietary providers you plan to use as environment variables (a `.env` file in the project root is also loaded automatically):

```bash
export GEMINI_API_KEY="your_key"      # google
export ANTHROPIC_API_KEY="your_key"   # anthropic
export OPENAI_API_KEY="your_key"      # openai
export XAI_API_KEY="your_key"         # xai
export MISTRAL_API_KEY="your_key"     # mistral
export DEEPSEEK_API_KEY="your_key"    # deepseek
```

Open-source providers (`ministral`, `llama`, `qwen`) do not require API keys, but may need a Hugging Face token for gated model downloads.

---

## 🧩 Versioned Prompts

Each task is driven by a structured prompt stored under `app/prompts/<task>/v<N>.txt`. Available versions are detected automatically, and each LLM command exposes a `--<task>_version` flag (defaulting to the latest version) so prompts can evolve without breaking reproducibility.

---

## 🧠 Recommended Workflow

```bash
# 1. Organize the dataset
python run.py prepare_dir ./books

# 2. OCR extraction
python run.py run_paddle ./books

# 3. Adaptive summarisation (run once per model)
python run.py resume ./books google gemini-2.5-flash 32000

# 4. Downstream tasks
python run.py abstract   ./books google gemini-2.5-flash
python run.py keywords   ./books google gemini-2.5-flash
python run.py categories ./books google gemini-2.5-flash
```

---

## 🧪 Tests

Unit tests live under `app/tests/` and can be run with:

```bash
pytest
```

---

## 📄 Citation

This multi-task version extends the original CompLiTT pipeline:

> L. H. da Silva Resende, R. P. Lopes, and A. D. Bravo, "CompLiTT: An end-to-end pipeline for processing and summarizing historical books," in *International Conference on Optimization, Learning Algorithms and Applications*, 2026 (in press).

The paper describing this extended version is currently under review; its citation will be added here once it is published.

Developed within the project *"CompLiTT — Computatio Litterarum limitis: Calculations of literary culture on the border"*, reference 2023.11359.PEX (DOI: 10.54499/2023.11359.PEX).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributions

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.
