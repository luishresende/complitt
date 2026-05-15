import logging
import os

from app.llms import LLMClient
from app.prompts import Prompt
from app.services.book import Book

logger = logging.getLogger(__name__)

_ABSTRACT_FILE = "abstract_contents.txt"
_SUMMARY_FILES = ("summary.txt", "summary_without_intro.txt")
_KEYWORDS_FILE = "keywords.txt"
_CATEGORIES_FILE = "categories.txt"


class BookAnalyzer:
    def __init__(
        self,
        book: Book,
        llm: LLMClient,
        summarize_prompt: Prompt,
        keywords_prompt: Prompt,
        categories_prompt: Prompt,
    ):
        self.book = book
        self.llm = llm
        self.summarize_prompt = summarize_prompt
        self.keywords_prompt = keywords_prompt
        self.categories_prompt = categories_prompt

    @property
    def _prefix(self) -> str:
        return f"{self.book.language}/{self.book.name}"

    def summarize(self, abstracts: list[str]) -> str:
        formatted = self.summarize_prompt.format(ABSTRACT_CONTENTS="\n\n".join(abstracts), LANGUAGE=self.book.language)
        return self.llm.generate_with_retry(formatted)

    def keywords(self, abstracts: list[str]) -> str:
        formatted = self.keywords_prompt.format(ABSTRACT_CONTENTS="\n\n".join(abstracts), LANGUAGE=self.book.language)
        return self.llm.generate_with_retry(formatted)

    def categories(self, abstracts: list[str]) -> str:
        formatted = self.categories_prompt.format(ABSTRACT_CONTENTS="\n\n".join(abstracts), LANGUAGE=self.book.language)
        return self.llm.generate_with_retry(formatted)

    def write_abstracts(self, abstracts: list[str]):
        """Persists abstract_contents.txt to disk."""
        if not abstracts:
            raise ValueError(f"[{self._prefix}] No abstracts to write — all splits failed")
        os.makedirs(self.book.provider_path, exist_ok=True)
        self._write_if_missing(_ABSTRACT_FILE, lambda: "\n\n".join(abstracts))

    def write_summary(self, abstracts: list[str]):
        """Generates and persists summary.txt and summary_without_intro.txt."""
        os.makedirs(self.book.provider_path, exist_ok=True)
        self._write_if_missing("summary.txt", lambda: self.summarize(abstracts))
        self._write_if_missing("summary_without_intro.txt", lambda: self.summarize(abstracts[1:]))

    def write_keywords(self, abstracts: list[str]):
        """Generates and persists keywords.txt."""
        os.makedirs(self.book.provider_path, exist_ok=True)
        self._write_if_missing(_KEYWORDS_FILE, lambda: self.keywords(abstracts))

    def write_categories(self, abstracts: list[str]):
        """Generates and persists categories.txt."""
        os.makedirs(self.book.provider_path, exist_ok=True)
        self._write_if_missing(_CATEGORIES_FILE, lambda: self.categories(abstracts))

    def abstracts_exist(self) -> bool:
        return self._file_has_content(os.path.join(self.book.provider_path, _ABSTRACT_FILE))

    def summary_exists(self) -> bool:
        return all(
            self._file_has_content(os.path.join(self.book.provider_path, name))
            for name in _SUMMARY_FILES
        )

    def keywords_exist(self) -> bool:
        return self._file_has_content(os.path.join(self.book.provider_path, _KEYWORDS_FILE))

    def categories_exist(self) -> bool:
        return self._file_has_content(os.path.join(self.book.provider_path, _CATEGORIES_FILE))

    def _file_has_content(self, path: str) -> bool:
        return os.path.exists(path) and os.path.getsize(path) > 0

    def _write_if_missing(self, filename: str, produce):
        path = os.path.join(self.book.provider_path, filename)
        if self._file_has_content(path):
            logger.info("[SKIP] %s already exists for %s", filename, self._prefix)
            return
        content = produce()
        with open(path, "w") as f:
            f.write(content)
