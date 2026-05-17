"""BERTScore + ROUGE evaluation of CompLiTT summaries.

For every book in the results dataset (`complitt_results`), each model's `summary.txt` is
compared against the human reference summary stored in `providers/human/
summary.txt`. BERTScore is computed with XLM-RoBERTa (multilingual); ROUGE-1/2/L
are computed with `rouge_score`.

The script is fully self-contained: it only needs the `complitt_results` results folder,
no external reference files.

Usage:
    python evaluation/bertscore_rouge.py \
        --books-dir /path/to/complitt_results \
        --output evaluation/results/bertscore_rouge.csv
"""

import argparse
import csv
from pathlib import Path

from bert_score import score as bertscore
from rouge_score import rouge_scorer

# Folders under complitt_results that are not languages.
SKIP_DIRS = {"do_not_use"}

# The provider folder holding the human-written reference summary.
REFERENCE_PROVIDER = "human"

# Maps a language folder name to the code BERTScore expects.
LANG_MAP = {
    "english": "en",
    "spanish": "es",
    "italian": "it",
}

SUMMARY_FILENAME = "summary.txt"

ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]


def find_summaries(providers_dir: Path) -> list[tuple[str, Path]]:
    """Return (provider_key, summary_path) for every provider under a book.

    Providers may be nested one level deep (e.g. `meta-llama/Llama-3.1-8B`);
    the provider key keeps the full relative path.
    """
    results: list[tuple[str, Path]] = []
    for entry in sorted(providers_dir.iterdir()):
        if not entry.is_dir():
            continue
        direct = entry / SUMMARY_FILENAME
        if direct.exists():
            results.append((entry.name, direct))
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and (sub / SUMMARY_FILENAME).exists():
                results.append((f"{entry.name}/{sub.name}", sub / SUMMARY_FILENAME))
    return results


def compute_rouge(candidate: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=True)
    s = scorer.score(reference, candidate)
    scores = {}
    for metric in ROUGE_METRICS:
        scores[f"{metric}_p"] = s[metric].precision
        scores[f"{metric}_r"] = s[metric].recall
        scores[f"{metric}_f"] = s[metric].fmeasure
    return scores


def compute_bertscore(candidate: str, reference: str, lang: str) -> dict:
    P, R, F1 = bertscore(
        [candidate],
        [reference],
        model_type="xlm-roberta-large",
        lang=lang,
        verbose=False,
    )
    return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}


def main():
    parser = argparse.ArgumentParser(description="BERTScore + ROUGE over complitt_results")
    parser.add_argument(
        "--books-dir",
        required=True,
        help="Path to the complitt_results results folder.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/bertscore_rouge.csv",
        help="CSV file to write the per-summary scores to.",
    )
    args = parser.parse_args()

    books_root = Path(args.books_dir)
    output_path = Path(args.output)
    if not books_root.is_dir():
        raise FileNotFoundError(f"Books directory not found: {books_root}")

    rows = []
    skipped = []

    lang_dirs = [
        d for d in sorted(books_root.iterdir())
        if d.is_dir() and d.name not in SKIP_DIRS
    ]

    for lang_dir in lang_dirs:
        lang_code = LANG_MAP.get(lang_dir.name.lower(), "en")
        book_dirs = [d for d in sorted(lang_dir.iterdir()) if d.is_dir()]

        for book_dir in book_dirs:
            providers_dir = book_dir / "providers"
            if not providers_dir.is_dir():
                continue

            summaries = find_summaries(providers_dir)
            reference_path = next(
                (p for key, p in summaries if key == REFERENCE_PROVIDER), None
            )
            if reference_path is None:
                print(f"[SKIP] No '{REFERENCE_PROVIDER}' reference for: {book_dir.name}")
                skipped.append(book_dir.name)
                continue

            reference_text = reference_path.read_text(encoding="utf-8").strip()
            if not reference_text:
                print(f"[SKIP] Empty reference for: {book_dir.name}")
                skipped.append(book_dir.name)
                continue

            print(f"[{lang_dir.name}] {book_dir.name}")

            for provider_key, summary_path in summaries:
                if provider_key == REFERENCE_PROVIDER:
                    continue
                candidate_text = summary_path.read_text(encoding="utf-8").strip()
                if not candidate_text:
                    continue

                bert = compute_bertscore(candidate_text, reference_text, lang_code)
                rouge = compute_rouge(candidate_text, reference_text)
                rows.append({
                    "language": lang_dir.name,
                    "book": book_dir.name,
                    "provider": provider_key,
                    "bert_precision": f"{bert['precision']:.4f}",
                    "bert_recall": f"{bert['recall']:.4f}",
                    "bert_f1": f"{bert['f1']:.4f}",
                    **{k: f"{v:.4f}" for k, v in rouge.items()},
                })
                print(
                    f"  {provider_key}: BERT F1={bert['f1']:.4f} | "
                    f"R1={rouge['rouge1_f']:.4f} R2={rouge['rouge2_f']:.4f} "
                    f"RL={rouge['rougeL_f']:.4f}"
                )

    fieldnames = [
        "language", "book", "provider",
        "bert_precision", "bert_recall", "bert_f1",
        "rouge1_p", "rouge1_r", "rouge1_f",
        "rouge2_p", "rouge2_r", "rouge2_f",
        "rougeL_p", "rougeL_r", "rougeL_f",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {output_path} ({len(rows)} rows)")
    if skipped:
        print(f"Skipped books ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()
