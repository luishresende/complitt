from app.utils import prepare_dir
from app.ocr import run_paddle
from app.postprocess import save_postprocess_output
from app.summarize import resume, keywords, abstract, categories
from app.prompts import Prompt

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="CompLiTT: A Summary Approach to Historical Books"
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # -----------------------------
    # prepare_dir
    # -----------------------------
    parser_prepare = subparsers.add_parser(
        "prepare_dir",
        help="Prepare the input books directory by creating folders for each PDF"
    )
    parser_prepare.add_argument(
        "input_dir",
        type=str,
        help="Path to the root directory containing book PDFs"
    )

    # -----------------------------
    # run_paddle
    # -----------------------------
    parser_run = subparsers.add_parser(
        "run_paddle",
        help="Run PaddleOCR for all books in the input directory"
    )
    parser_run.add_argument(
        "input_dir",
        type=str,
        help="Input directory prepared with 'prepare_dir'"
    )
    parser_run.add_argument(
        "--output_relative_dir",
        type=str,
        default="paddle_output",
        help="Relative folder name to save PaddleOCR results for each book (default: 'paddle_output')"
    )

    # -----------------------------
    # post_process_paddle_output
    # -----------------------------
    parser_post = subparsers.add_parser(
        "post_process_paddle",
        help="Post-process PaddleOCR output and save results in the book directories. This step is executed during summarization and is optional."
    )
    parser_post.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing books"
    )
    parser_post.add_argument(
        "--paddle_output_relative_dir",
        type=str,
        default="paddle_output",
        help="Relative dir where PaddleOCR results are stored (default: 'paddle_output')"
    )
    parser_post.add_argument(
        "--save_output_relative_path",
        type=str,
        default="book_content.md",
        help="Relative path to save post-processed output for each book (default: 'book_content.md')"
    )

    # Shared arguments for LLM-based commands
    def add_llm_args(p):
        p.add_argument("input_dir", type=str, help="Root directory containing books")
        p.add_argument("api", type=str, choices=["google", "anthropic", "openai", "xai", "mistral", "deepseek", "ministral", "llama", "qwen"], help="API to use")
        p.add_argument("model", type=str, help="Model name to use with the selected API")
        p.add_argument(
            "--max_retries",
            type=int,
            default=-1,
            help="Max retries on transient API errors (rate limit, server errors). -1 = infinite. Only applies to paid APIs (google, anthropic, openai, xai, mistral, deepseek). (default: -1)"
        )

    # -----------------------------
    # resume
    # -----------------------------
    parser_resume = subparsers.add_parser(
        "resume",
        help="Compress book chunks into intermediate abstracts (required before 'abstract' and 'keywords')"
    )
    add_llm_args(parser_resume)
    parser_resume.add_argument(
        "context_window_size",
        type=int,
        help="Context window size (in characters) of the model"
    )
    parser_resume.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads for parallel split processing (default: 1)"
    )
    parser_resume.add_argument(
        "--paddle_output_relative_dir",
        type=str,
        default="paddle_output",
        help="Relative dir where PaddleOCR results are stored (default: 'paddle_output')"
    )
    _resume_versions = Prompt.available_versions("resume")
    parser_resume.add_argument(
        "--resume_version",
        type=int,
        default=_resume_versions[-1],
        choices=_resume_versions,
        help=f"Version of the resume prompt (default: {_resume_versions[-1]})"
    )

    # -----------------------------
    # abstract
    # -----------------------------
    parser_abstract = subparsers.add_parser(
        "abstract",
        help="Generate the final book abstract from previously generated abstracts"
    )
    add_llm_args(parser_abstract)
    _abstract_versions = Prompt.available_versions("abstract")
    parser_abstract.add_argument(
        "--abstract_version",
        type=int,
        default=_abstract_versions[-1],
        choices=_abstract_versions,
        help=f"Version of the abstract prompt (default: {_abstract_versions[-1]})"
    )

    # -----------------------------
    # keywords
    # -----------------------------
    parser_keywords = subparsers.add_parser(
        "keywords",
        help="Extract keywords/themes from previously generated abstracts"
    )
    add_llm_args(parser_keywords)
    _keywords_versions = Prompt.available_versions("keywords")
    parser_keywords.add_argument(
        "--keywords_version",
        type=int,
        default=_keywords_versions[-1],
        choices=_keywords_versions,
        help=f"Version of the keywords prompt (default: {_keywords_versions[-1]})"
    )

    # -----------------------------
    # categories
    # -----------------------------
    parser_categories = subparsers.add_parser(
        "categories",
        help="Classify books into academic disciplines/knowledge areas from previously generated abstracts"
    )
    add_llm_args(parser_categories)
    _categories_versions = Prompt.available_versions("categories")
    parser_categories.add_argument(
        "--categories_version",
        type=int,
        default=_categories_versions[-1],
        choices=_categories_versions,
        help=f"Version of the categories prompt (default: {_categories_versions[-1]})"
    )

    args = parser.parse_args()

    # -----------------------------
    # Dispatch to functions
    # -----------------------------
    if args.command == "prepare_dir":
        prepare_dir(args.input_dir)

    elif args.command == "run_paddle":
        run_paddle(args.input_dir, relative_output_dir=args.output_relative_dir)

    elif args.command == "post_process_paddle":
        save_postprocess_output(
            args.input_dir,
            paddle_output_relative_dir=args.paddle_output_relative_dir,
            save_output_relative_path=args.save_output_relative_path
        )

    elif args.command == "resume":
        resume(
            args.input_dir,
            api=args.api,
            model=args.model,
            context_window_size=args.context_window_size,
            resume_version=args.resume_version,
            num_threads=args.num_threads,
            paddle_output_relative_dir=args.paddle_output_relative_dir,
            max_retries=args.max_retries,
        )

    elif args.command == "abstract":
        abstract(
            args.input_dir,
            api=args.api,
            model=args.model,
            abstract_version=args.abstract_version,
            max_retries=args.max_retries,
        )

    elif args.command == "keywords":
        keywords(
            args.input_dir,
            api=args.api,
            model=args.model,
            keywords_version=args.keywords_version,
            max_retries=args.max_retries,
        )

    elif args.command == "categories":
        categories(
            args.input_dir,
            api=args.api,
            model=args.model,
            categories_version=args.categories_version,
            max_retries=args.max_retries,
        )

if __name__ == "__main__":
    main()
