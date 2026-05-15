import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from app.llms import LLMClient
from app.prompts import Prompt
from app.services.book import Book

logger = logging.getLogger(__name__)


class BookResumer:
    def __init__(
        self,
        book: Book,
        llm: LLMClient,
        context_window_size: int,
        resume_prompt: Prompt,
        num_threads: int = 1,
    ):
        self.book = book
        self.llm = llm
        self.context_window_size = context_window_size
        self.resume_prompt = resume_prompt
        self.num_threads = num_threads
        self.max_chunk_size = context_window_size - resume_prompt.overhead

    @property
    def _prefix(self) -> str:
        return f"{self.book.language}/{self.book.name}"

    @staticmethod
    def build_splits(content: str, max_chunk_size: int) -> list[tuple[int, int]]:
        initial_index = 0
        splits = []
        while initial_index < len(content):
            search_start = initial_index + max_chunk_size
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

    def _process_split(self, split: dict, content: str):
        if split["content_file"]:
            return
        chunk = content[split["initial_pos"]:split["end_pos"]]
        target_size = int(len(chunk) * 0.10)

        formatted = self.resume_prompt.format(TARGET_SIZE=target_size, TEXT=chunk, LANGUAGE=self.book.language)
        response = self.llm.generate_with_retry(formatted)
        self.book.save_split_abstract(split, response)

    def _run_batch(self, content: str):
        self.book.reload()
        batch_num = len(self.book.batches)
        logger.info("[%s] Processing batch %d (%d chars)", self._prefix, batch_num, len(content))

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_split, split, content): split["index"]
                for split in self.book.current_batch
            }
            with tqdm(total=len(futures), desc=f"{self._prefix} batch {batch_num}") as pbar:
                for future in as_completed(futures):
                    if exc := future.exception():
                        logger.error("[%s] Error on split %s: %s", self._prefix, futures[future], exc)
                    pbar.update(1)

    def run(self, content: str) -> list[str]:
        """Iteratively summarizes content until it fits the context window. Returns the final abstracts."""
        while True:
            self._run_batch(content)

            self.book.reload()
            abstracts = self.book.load_abstracts()
            content = "\n\n".join(abstracts)

            if len(content) <= self.context_window_size:
                break

            self.book.add_batch(BookResumer.build_splits(content, self.max_chunk_size))

        return abstracts
