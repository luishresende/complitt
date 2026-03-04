from app.utils import prepare_dir
from app.ocr import run_paddle
from app.postprocess import save_postprocess_output
from app.summarize import summarize

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

    # -----------------------------
    # summarize
    # -----------------------------
    parser_summarize = subparsers.add_parser(
        "summarize",
        help="Summarize the content of all books in the input directory using a selected API model"
    )
    parser_summarize.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing books to summarize"
    )
    parser_summarize.add_argument(
        "api",
        type=str,
        choices=["google", "anthropic", "openai", "xai"],
        help="API to use for summarization"
    )
    parser_summarize.add_argument(
        "model",
        type=str,
        help="Model name to use for summarization with the selected API"
    )
    parser_summarize.add_argument(
        "--paddle_output_relative_dir",
        type=str,
        default="paddle_output",
        help="Relative dir where PaddleOCR results are stored (default: 'paddle_output')"
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

    elif args.command == "summarize":
        summarize(
            args.input_dir,
            api=args.api,
            model=args.model,
            paddle_output_relative_dir=args.paddle_output_relative_dir
        )

if __name__ == "__main__":
    main()