from app.preprocess import fix_content_pipeline, adjust_text, replace_divs, get_md_files, merge_md_files, replace_tables
from app.services.book import (
    create_book, get_book,
    generate_abstract, summarize_abstracts, load_abstracts, _book_provider_path, _meta_path
)
from app.llms import LLMClient
from app.llms.gemini_client import GeminiClient
from app.llms.openai_client import OpenAIClient
from app.llms.claude_provider import ClaudeClient
from app.llms.xai_client import XAIClient

from dotenv import load_dotenv
from tqdm import tqdm

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

CHAR_PER_PAGE = 2000
MINIMUM_CHARS_PER_PAGE = CHAR_PER_PAGE * 2
BOOKS_ROOT = "books"


def get_md_content(input_path):
    if not os.path.exists(input_path):
        raise ValueError("Input does not exist")

    if os.path.isfile(input_path) and os.path.splitext(input_path)[1] == ".md":
        with open(input_path, "r") as f:
            md_content = f.read()
        md_content = fix_content_pipeline(md_content, replace_divs, replace_tables)

    elif os.path.isdir(input_path):
        md_files = get_md_files(input_path)
        if not md_files:
            raise ValueError("Input is not a markdown file or a path with markdown files")

        md_content = merge_md_files(md_files)
        with open(os.path.join(input_path, 'merged_debug.md'), "w") as f:
            f.write(md_content)
        md_content = fix_content_pipeline(md_content, replace_divs, adjust_text, replace_tables)

    return md_content


def build_splits(content: str) -> list[tuple[int, int]]:
    initial_index = 0
    splits = []
    while initial_index < len(content):
        search_start = initial_index + MINIMUM_CHARS_PER_PAGE
        if search_start >= len(content):
            splits.append((initial_index, len(content)))
            break
        newline_pos = content.find("\n", search_start)
        if newline_pos == -1:
            splits.append((initial_index, len(content)))
            break
        end_index = newline_pos + 1
        splits.append((initial_index, end_index))
        initial_index = end_index
    return splits


from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_split(split, content, book, llm, prefix):
    if split["content_file"]:
        # logger.info("[%s] Abstract for split %s already generated.", prefix, split['index'])
        return
    # logger.info("[%s] Generating abstract for split %s: (%s, %s)",
    #             prefix, split['index'], split['initial_pos'], split['end_pos'])
    generate_abstract(content, book, split, llm)


def process_book(book_name: str, language: str, book_input_path: str, llm: LLMClient, num_threads: int = 1):
    prefix = f"{language}/{book_name}"
    book_dir = os.path.join(BOOKS_ROOT, language, book_name)
    output_book_dir = _book_provider_path(book_name, language, llm.name)
    output_abstracts_path = os.path.join(output_book_dir, "abstract_contents.txt")
    output_summary_path = os.path.join(output_book_dir, "summary.txt")
    output_summary_without_intro_path = os.path.join(output_book_dir, "summary_without_intro.txt")

    if all([
        os.path.exists(output_abstracts_path),
        os.path.exists(output_summary_path),
        os.path.exists(output_summary_without_intro_path),
    ]):
        logger.info("[SKIP] %s já existe para o provider '%s'.", prefix, llm.name)
        return

    logger.info("[PROCESS] %s", prefix)

    paddle_output_path = os.path.join(book_dir, "paddle_output")
    if not os.path.exists(paddle_output_path):
        logger.warning("[SKIP] No paddle output found for %s", prefix)
        return

    content = get_md_content(paddle_output_path)
    splits = build_splits(content)

    book = create_book(book_name, language, splits, llm.name)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(_process_split, split, content, book, llm, prefix): split['index']
            for split in book["splits"]
        }
        with tqdm(total=len(futures), desc=f"{prefix} splits") as pbar:
            for future in as_completed(futures):
                if exc := future.exception():
                    logger.error("[%s] Error on split %s: %s", prefix, futures[future], exc)
                pbar.update(1)

    book = get_book(book_name, language, llm.name)
    abstracts = load_abstracts(book)

    os.makedirs(output_book_dir, exist_ok=True)
    if not os.path.exists(output_abstracts_path):
        with open(output_abstracts_path, "w") as f:
            f.write("\n\n".join(abstracts))
    else:
        logger.info("[SKIP] Abstracts already generated for %s", prefix)

    if not os.path.exists(output_summary_path):
        with open(output_summary_path, "w") as f:
            f.write(summarize_abstracts(abstracts, llm))
    else:
        logger.info("[SKIP] Summary already generated for %s", prefix)

    if not os.path.exists(output_summary_without_intro_path):
        with open(output_summary_without_intro_path, "w") as f:
            f.write(summarize_abstracts(abstracts[1:], llm))
    else:
        logger.info("[SKIP] Summary without intro already generated for %s", prefix)

    logger.info("[DONE] %s", prefix)


def main():
    llm_provider = XAIClient("grok-4-fast-reasoning")

    for language in os.listdir(BOOKS_ROOT):
        language_path = os.path.join(BOOKS_ROOT, language)
        if not os.path.isdir(language_path):
            continue

        for book_name in os.listdir(language_path):
            book_input_path = os.path.join(language_path, book_name)
            if not os.path.isdir(book_input_path):
                continue

            try:
                process_book(book_name, language, book_input_path, llm_provider, num_threads=5)
            except Exception as e:
                print(f"[ERROR] {language}/{book_name}: {e}")

if __name__ == "__main__":
    main()
