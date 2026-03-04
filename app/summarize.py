from app.postprocess import get_md_content
from app.services.book import (
    create_book, get_book,
    generate_abstract, summarize_abstracts, load_abstracts, _book_provider_path
)
from app.llms import LLMClient
from app.llms.gemini_client import GeminiClient
from app.llms.openai_client import OpenAIClient
from app.llms.claude_provider import ClaudeClient
from app.llms.xai_client import XAIClient

from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _process_split(split, content, book, llm, prefix):
    if split["content_file"]:
        return
    generate_abstract(content, book, split, llm)


def process_book(books_root: str, book_name: str, language: str, llm: LLMClient, num_threads: int = 1, paddle_relative_output_dir: str = "paddle_output"):
    prefix = f"{language}/{book_name}"
    book_dir = os.path.join(books_root, language, book_name)
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

    paddle_output_path = os.path.join(book_dir, paddle_relative_output_dir)
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


def summarize(books_root, api, model, num_threads=1, paddle_output_relative_dir="paddle_output"):
    if api == "openai":
        llm_provider = OpenAIClient(model)
    elif api == "anthropic":
        llm_provider = ClaudeClient(model)
    elif api == "gemini":
        llm_provider = GeminiClient(model)
    elif api == "xai":
        llm_provider = XAIClient(model)
    else:
        raise ValueError(f"API {api} not supported, custom implementation required.")

    for language in os.listdir(books_root):
        language_path = os.path.join(books_root, language)
        if not os.path.isdir(language_path):
            continue

        for book_name in os.listdir(language_path):
            book_input_path = os.path.join(language_path, book_name)
            if not os.path.isdir(book_input_path):
                continue

            try:
                process_book(books_root, book_name, language, llm_provider, num_threads=num_threads, paddle_relative_output_dir=paddle_output_relative_dir)
            except Exception as e:
                print(f"[ERROR] {language}/{book_name}: {e}")

