# CompLiTT — A Summary Approach to Historical Books

CompLiTT is a command-line pipeline designed to process historical books in PDF format and generate structured summaries using modern OCR and Large Language Models (LLMs). You can check our results [here](https://drive.google.com/drive/folders/1mMKY8ry4GjUa8ofzVkv5VJKde4sF3D_A?usp=drive_link).

The workflow is divided into **three sequential stages**:

1. Prepare the dataset structure
2. Extract text using OCR (PaddleOCR)
3. Generate summaries using LLM APIs

---

## 📦 Overview

CompLiTT converts collections of historical PDFs into summarized content through the following pipeline:

```
PDFs → Organized Dataset → OCR → Clean Text → LLM Summaries
```

Each step is exposed as a CLI command.

---

## 📁 Expected Directory Structure

Books must be organized by **language**, with one folder per language.

Language names are arbitrary.

```
input_dir/
├── portuguese/
│   ├── book_a.pdf
│   └── book_b.pdf
├── spanish/
│   └── book_c.pdf
```

CompLiTT enforces this hierarchy:

```
language → book → pdf
```

---

# ⚙️ Installation

Clone the repository:

``` bash
git clone <repo-url>
cd compliTT
```

Install project dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Install PaddleOCR

CompLiTT relies on **PaddleOCR** for text extraction.\
You must install PaddlePaddle and PaddleOCR following the official
instructions, since installation depends on your system configuration
(CPU/GPU/CUDA version).

👉 Official guide: https://github.com/PaddlePaddle/PaddleOCR

------------------------------------------------------------------------

## Requirements

Ensure your environment has:

-   Python ≥ 3.10
-   CUDA (optional, for GPU OCR acceleration)
-   API keys configured for your chosen LLM provider

---

## 🚀 Usage

All commands are executed through:

```bash
python run.py <command> [arguments]
```

---

# 1️⃣ Prepare Directory

Organizes PDFs into the required structure by creating one folder per book and moving each PDF into its corresponding directory.

### Command

```bash
python run.py prepare_dir <input_dir>
```

### Example

```bash
python run.py prepare_dir ./books
```

### What it does

* Traverses language folders
* Creates a folder for each PDF
* Moves PDFs into matching folders
* Skips already organized books

---

# 2️⃣ Run OCR (PaddleOCR)

Extracts text and layout information from all books using PaddleOCR.

GPU is automatically used if available.

### Command

```bash
python run.py run_paddle <input_dir>
```

### Example

```bash
python run.py run_paddle ./books
```

### Output

For each book:

```
book/
├── book.pdf
└── paddle_output/
    ├── *.json
    └── *.md
```

---

# 3️⃣ Post-Process OCR Output (Optional)

Cleans and merges OCR results into a single Markdown file per book.

⚠️ This step is **optional** because it is automatically executed during summarization.

### Command

```bash
python run.py post_process_paddle <input_dir>
```

### Example

```bash
python run.py post_process_paddle ./books
```

### Output

```
book/
├── book.pdf
├── paddle_output/
└── book_content.md
```

---

# 4️⃣ Summarize Books

Generates summaries using an LLM API.

Supported providers:

* `google`
* `anthropic`
* `openai`
* `xai`

### Command

```bash
python run.py summarize <input_dir> <api> <model>
```

### Example

```bash
python run.py summarize ./books openai gpt-4.1
```

---

## 🔑 API Configuration

Set your API keys using environment variables.

Example:

```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
export XAI_API_KEY="your_key"
```

---

## 🧠 Recommended Workflow

Typical execution order:

```bash
# 1. Organize dataset
python run.py prepare_dir ./books

# 2. OCR extraction
python run.py run_paddle ./books

# 3. (Optional) Inspect processed text
python run.py post_process_paddle ./books

# 4. Generate summaries
python run.py summarize ./books openai gpt-4.1
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributions

Contributions, bug reports, and feature requests are welcome.
Please open an issue or submit a pull request.

---
