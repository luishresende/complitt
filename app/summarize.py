from app.postprocess import get_md_content
from app.prompts import Prompt
from app.services.book import Book
from app.services.summarizer import BookResumer
from app.services.analyzer import BookAnalyzer
from app.llms import LLMClient
from app.llms.gemini_client import GeminiClient
from app.llms.openai_client import OpenAIClient
from app.llms.claude_client import ClaudeClient
from app.llms.xai_client import XAIClient
from app.llms.mistral_client import MistralClient
from app.llms.deepseek_client import DeepSeekClient
from app.llms.ministral_client import MinistralClient
from app.llms.llama_client import LlamaClient
from app.llms.qwen_client import QwenClient

from dotenv import load_dotenv

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

def _load_prompts(resume_version: int = 1, abstract_version: int = 1, keywords_version: int = 1, categories_version: int = 1):
    return (
        Prompt.load("resume", version=resume_version, keys=["TARGET_SIZE", "TEXT", "LANGUAGE"]),
        Prompt.load("abstract", version=abstract_version, keys=["ABSTRACT_CONTENTS", "LANGUAGE"]),
        Prompt.load("keywords", version=keywords_version, keys=["ABSTRACT_CONTENTS", "LANGUAGE"]),
        Prompt.load("categories", version=categories_version, keys=["ABSTRACT_CONTENTS", "LANGUAGE"]),
    )

_LLM_PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": ClaudeClient,
    "google": GeminiClient,
    "xai": XAIClient,
    "mistral": MistralClient,
    "deepseek": DeepSeekClient,
    "ministral": MinistralClient,
    "llama": LlamaClient,
    "qwen": QwenClient,
}


def _build_llm(api: str, model: str) -> LLMClient:
    client_cls = _LLM_PROVIDERS.get(api)
    if client_cls is None:
        raise ValueError(f"API '{api}' not supported. Choose from: {list(_LLM_PROVIDERS)}")
    return client_cls(model)


def _iter_books(books_root: str):
    """Yields (language, book_name) for all books in the root directory."""
    for language in os.listdir(books_root):
        language_path = os.path.join(books_root, language)
        if not os.path.isdir(language_path):
            continue
        for book_name in os.listdir(language_path):
            if os.path.isdir(os.path.join(language_path, book_name)):
                yield language, book_name


def process_book(books_root: str, book_name: str, language: str, llm: LLMClient, context_window_size: int, resume_prompt: Prompt, summarize_prompt: Prompt, keywords_prompt: Prompt, categories_prompt: Prompt, num_threads: int = 1, paddle_relative_output_dir: str = "paddle_output"):
    """Runs the hierarchical summarization pipeline and saves the intermediate abstracts."""
    prefix = f"{language}/{book_name}"
    book_path = os.path.join(books_root, language, book_name)

    try:
        content = get_md_content(book_path, paddle_relative_output_dir)
    except (ValueError, FileNotFoundError):
        logger.warning("[SKIP] No paddle output found for %s", prefix)
        return

    try:
        book = Book.load(book_name, language, llm.name, books_root)
    except FileNotFoundError:
        max_chunk_size = context_window_size - resume_prompt.overhead
        book = Book.create(book_name, language, BookResumer.build_splits(content, max_chunk_size), llm.name, books_root)

    analyzer = BookAnalyzer(book, llm, summarize_prompt, keywords_prompt, categories_prompt)

    if analyzer.abstracts_exist():
        logger.info("[SKIP] %s already resumed for provider '%s'.", prefix, llm.name)
        return

    resumer = BookResumer(book, llm, context_window_size, resume_prompt, num_threads)

    logger.info("[RESUME] %s", prefix)
    abstracts = resumer.run(content)
    analyzer.write_abstracts(abstracts)
    logger.info("[DONE] %s", prefix)


def process_book_keywords(book_name: str, language: str, llm: LLMClient, abstract_prompt: Prompt, keywords_prompt: Prompt, categories_prompt: Prompt, books_root: str = "books"):
    """Loads existing abstracts and generates keywords.txt."""
    prefix = f"{language}/{book_name}"

    try:
        book = Book.load(book_name, language, llm.name, books_root)
    except FileNotFoundError:
        logger.warning("[SKIP] %s — no resume found for provider '%s'. Run 'resume' first.", prefix, llm.name)
        return

    analyzer = BookAnalyzer(book, llm, abstract_prompt, keywords_prompt, categories_prompt)

    if not analyzer.abstracts_exist():
        logger.warning("[SKIP] %s — abstracts not found for provider '%s'. Run 'resume' first.", prefix, llm.name)
        return

    if analyzer.keywords_exist():
        logger.info("[SKIP] %s — keywords already exist for provider '%s'.", prefix, llm.name)
        return

    abstracts = book.load_abstracts()
    logger.info("[KEYWORDS] %s", prefix)
    analyzer.write_keywords(abstracts)
    logger.info("[DONE] %s", prefix)


def process_book_abstract(book_name: str, language: str, llm: LLMClient, abstract_prompt: Prompt, keywords_prompt: Prompt, categories_prompt: Prompt, books_root: str = "books"):
    """Loads existing abstracts and generates summary.txt and summary_without_intro.txt."""
    prefix = f"{language}/{book_name}"

    try:
        book = Book.load(book_name, language, llm.name, books_root)
    except FileNotFoundError:
        logger.warning("[SKIP] %s — no resume found for provider '%s'. Run 'resume' first.", prefix, llm.name)
        return

    analyzer = BookAnalyzer(book, llm, abstract_prompt, keywords_prompt, categories_prompt)

    if not analyzer.abstracts_exist():
        logger.warning("[SKIP] %s — abstracts not found for provider '%s'. Run 'resume' first.", prefix, llm.name)
        return

    if analyzer.summary_exists():
        logger.info("[SKIP] %s — abstract already exists for provider '%s'.", prefix, llm.name)
        return

    abstracts = book.load_abstracts()
    logger.info("[ABSTRACT] %s", prefix)
    analyzer.write_summary(abstracts)
    logger.info("[DONE] %s", prefix)


def process_book_categories(book_name: str, language: str, llm: LLMClient, abstract_prompt: Prompt, keywords_prompt: Prompt, categories_prompt: Prompt, books_root: str = "books"):
    """Loads existing abstracts and generates categories.txt."""
    prefix = f"{language}/{book_name}"

    try:
        book = Book.load(book_name, language, llm.name, books_root)
    except FileNotFoundError:
        logger.warning("[SKIP] %s — no resume found for provider '%s'. Run 'resume' first.", prefix, llm.name)
        return

    analyzer = BookAnalyzer(book, llm, abstract_prompt, keywords_prompt, categories_prompt)

    if not analyzer.abstracts_exist():
        logger.warning("[SKIP] %s — abstracts not found for provider '%s'. Run 'resume' first.", prefix, llm.name)
        return

    if analyzer.categories_exist():
        logger.info("[SKIP] %s — categories already exist for provider '%s'.", prefix, llm.name)
        return

    abstracts = book.load_abstracts()
    logger.info("[CATEGORIES] %s", prefix)
    analyzer.write_categories(abstracts)
    logger.info("[DONE] %s", prefix)


def resume(books_root, api, model, context_window_size, resume_version=1, num_threads=1, paddle_output_relative_dir="paddle_output", max_retries=-1):
    llm = _build_llm(api, model)
    llm.default_max_retries = max_retries
    resume_prompt, abstract_prompt, keywords_prompt, categories_prompt = _load_prompts(resume_version=resume_version)
    for language, book_name in _iter_books(books_root):
        try:
            process_book(books_root, book_name, language, llm, context_window_size, resume_prompt, abstract_prompt, keywords_prompt, categories_prompt, num_threads=num_threads, paddle_relative_output_dir=paddle_output_relative_dir)
        except Exception as e:
            logger.error("%s/%s: %s", language, book_name, e)


def keywords(books_root, api, model, keywords_version=1, max_retries=-1):
    llm = _build_llm(api, model)
    llm.default_max_retries = max_retries
    _, abstract_prompt, keywords_prompt, categories_prompt = _load_prompts(keywords_version=keywords_version)
    for language, book_name in _iter_books(books_root):
        try:
            process_book_keywords(book_name, language, llm, abstract_prompt, keywords_prompt, categories_prompt, books_root)
        except Exception as e:
            logger.error("%s/%s: %s", language, book_name, e)


def abstract(books_root, api, model, abstract_version=1, max_retries=-1):
    llm = _build_llm(api, model)
    llm.default_max_retries = max_retries
    _, abstract_prompt, keywords_prompt, categories_prompt = _load_prompts(abstract_version=abstract_version)
    for language, book_name in _iter_books(books_root):
        try:
            process_book_abstract(book_name, language, llm, abstract_prompt, keywords_prompt, categories_prompt, books_root)
        except Exception as e:
            logger.error("%s/%s: %s", language, book_name, e)


def categories(books_root, api, model, categories_version=1, max_retries=-1):
    llm = _build_llm(api, model)
    llm.default_max_retries = max_retries
    _, abstract_prompt, keywords_prompt, categories_prompt = _load_prompts(categories_version=categories_version)
    for language, book_name in _iter_books(books_root):
        try:
            process_book_categories(book_name, language, llm, abstract_prompt, keywords_prompt, categories_prompt, books_root)
        except Exception as e:
            logger.error("%s/%s: %s", language, book_name, e)