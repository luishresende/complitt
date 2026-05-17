"""BooookScore evaluation of CompLiTT summaries.

BooookScore measures the *coherence* of a summary on its own (it needs no
reference): an LLM annotator inspects every sentence and flags the ones that
introduce confusion. The score is the fraction of non-confusing sentences,
averaged over books.

This wrapper:
  1. discovers every provider in the `complitt_results` results folder;
  2. collects each provider's `summary.txt` into a `summ_<provider>.json` file;
  3. runs BooookScore's annotator with the chosen evaluator model;
  4. writes a `booookscore_results.csv` with one score per provider.

Steps 1-2 are free and offline. Steps 3-4 call a paid LLM API — use
`--summ-only` to stop after step 2.

BooookScore itself is NOT vendored here: clone the (modified) fork first, e.g.

    git clone https://github.com/luishresende/BooookScore.git evaluation/BooookScore
    pip install -r evaluation/BooookScore/requirements.txt

Usage:
    # offline: only build the summ_*.json files
    python evaluation/booookscore_eval.py \
        --books-dir /path/to/complitt_results --output-dir evaluation/results/booookscore --summ-only

    # full run (calls the evaluator API)
    python evaluation/booookscore_eval.py \
        --books-dir /path/to/complitt_results --output-dir evaluation/results/booookscore \
        --api anthropic --model claude-sonnet-4-6
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SKIP_DIRS = {"do_not_use"}
SUMMARY_FILENAME = "summary.txt"

# Default location of the cloned BooookScore fork (kept out of git via
# the repo's .gitignore entry for `BooookScore/`).
DEFAULT_BOOOOKSCORE_DIR = Path(__file__).resolve().parent / "BooookScore"


def clean_summary(text: str) -> str:
    """Strip the boilerplate intro / trailing keyword section from a summary."""
    text = text.replace("Resumo do livro:\n", "")
    text = text.replace("Book Summary:\n", "")
    keyword_index = text.find("Palavras-chave")
    if keyword_index == -1:
        keyword_index = text.find("Keywords")
    if keyword_index != -1:
        text = text[:keyword_index]
    return text.strip()


def discover_providers(books_dir: Path) -> dict[str, str]:
    """Map a short provider name to the `summary.txt` path relative to `providers/`.

    Providers may be nested one level deep; the short name is the top-level
    folder (e.g. `meta-llama/Llama-3.1-8B-Instruct` -> `meta-llama`).
    """
    providers: dict[str, str] = {}
    for lang_dir in sorted(books_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name in SKIP_DIRS:
            continue
        for book_dir in sorted(lang_dir.iterdir()):
            providers_dir = book_dir / "providers"
            if not providers_dir.is_dir():
                continue
            for entry in sorted(providers_dir.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / SUMMARY_FILENAME).exists():
                    providers.setdefault(entry.name, entry.name)
                    continue
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir() and (sub / SUMMARY_FILENAME).exists():
                        providers.setdefault(entry.name, f"{entry.name}/{sub.name}")
    return providers


def build_summ_file(books_dir: Path, provider_relpath: str, output_path: Path) -> int:
    """Collect a provider's summaries into a {book: summary} JSON file."""
    summ_data: dict[str, str] = {}
    for lang_dir in sorted(books_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name in SKIP_DIRS:
            continue
        for book_dir in sorted(lang_dir.iterdir()):
            if not book_dir.is_dir():
                continue
            summary_path = book_dir / "providers" / provider_relpath / SUMMARY_FILENAME
            if not summary_path.exists():
                print(f"  [skip] missing {summary_path}")
                continue
            summary = clean_summary(summary_path.read_text(encoding="utf-8"))
            summ_data[book_dir.name] = summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summ_data, f, indent=4, ensure_ascii=False)
    return len(summ_data)


def ensure_nltk():
    """BooookScore tokenises summaries into sentences with NLTK."""
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def main():
    parser = argparse.ArgumentParser(description="BooookScore over complitt_results")
    parser.add_argument("--books-dir", required=True, help="Path to the complitt_results results folder.")
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/booookscore",
        help="Directory for the summ_*.json, annot_*.json and results CSV.",
    )
    parser.add_argument(
        "--booookscore-dir",
        default=str(DEFAULT_BOOOOKSCORE_DIR),
        help="Path to the cloned BooookScore fork (default: evaluation/BooookScore).",
    )
    parser.add_argument("--api", default="anthropic",
                        help="Evaluator provider (anthropic, openai, gemini, xai).")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Evaluator model name.")
    parser.add_argument("--providers", nargs="*", default=None,
                        help="Subset of provider names to score (default: all discovered).")
    parser.add_argument("--summ-only", action="store_true",
                        help="Only build the summ_*.json files; do not call any API.")
    args = parser.parse_args()

    books_dir = Path(args.books_dir)
    output_dir = Path(args.output_dir)
    if not books_dir.is_dir():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_providers(books_dir)
    if not discovered:
        raise RuntimeError(f"No providers with a {SUMMARY_FILENAME} found under {books_dir}")

    selected = args.providers or sorted(discovered)
    unknown = [p for p in selected if p not in discovered]
    if unknown:
        raise ValueError(f"Unknown providers {unknown}; available: {sorted(discovered)}")

    print(f"Providers to evaluate: {selected}")

    # Step 1-2: build the summ_*.json files.
    summ_paths: dict[str, Path] = {}
    for provider in selected:
        summ_path = output_dir / f"summ_{provider}.json"
        count = build_summ_file(books_dir, discovered[provider], summ_path)
        summ_paths[provider] = summ_path
        print(f"  {provider}: {count} summaries -> {summ_path}")

    if args.summ_only:
        print("\n--summ-only set; stopping before the API calls.")
        return

    # Locate the cloned BooookScore fork and make its package importable.
    booookscore_dir = Path(args.booookscore_dir)
    if not (booookscore_dir / "booookscore").is_dir():
        raise FileNotFoundError(
            f"BooookScore package not found in {booookscore_dir}. Clone the fork first:\n"
            f"  git clone https://github.com/luishresende/BooookScore.git {booookscore_dir}"
        )
    sys.path.insert(0, str(booookscore_dir))
    from booookscore.score import Scorer  # noqa: E402

    template_path = booookscore_dir / "prompts" / "get_annotations.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Annotation prompt not found: {template_path}")

    ensure_nltk()

    # Step 3-4: run the annotator and collect the scores. The evaluator client
    # reads its key from the environment (a .env file is loaded automatically).
    results = []
    for provider in selected:
        annot_path = output_dir / f"annot_{provider}_{args.model}.json"
        scorer = Scorer(
            model=args.model,
            api=args.api,
            summ_path=str(summ_paths[provider]),
            annot_path=str(annot_path),
            template_path=str(template_path),
            v2=False,
            batch_size=0,
        )
        score = scorer.get_score()
        results.append({"provider": provider, "booookscore": score})
        print(f"BooookScore {provider} = {score}")

    results_csv = output_dir / "booookscore_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["provider", "booookscore"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
