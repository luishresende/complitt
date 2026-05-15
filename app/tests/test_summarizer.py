import pytest
from unittest.mock import MagicMock, call, patch

from app.services.summarizer import BookResumer


@pytest.fixture
def mock_book(tmp_path):
    book = MagicMock()
    book.language = "en"
    book.name = "testbook"
    book.provider_path = str(tmp_path)
    return book


@pytest.fixture
def mock_resume_prompt():
    prompt = MagicMock()
    prompt.overhead = 100
    prompt.format.side_effect = lambda **kw: f"prompt:{kw['TEXT']}"
    return prompt


@pytest.fixture
def resumer(mock_book, mock_resume_prompt):
    mock_llm = MagicMock()
    mock_llm.generate_with_retry.return_value = "summary"
    return BookResumer(mock_book, mock_llm, context_window_size=1000, resume_prompt=mock_resume_prompt)


# --- helpers ---

def make_split(index: int, start: int, end: int, done: bool = False) -> dict:
    return {"index": index, "initial_pos": start, "end_pos": end, "content_file": "f.txt" if done else None}


class TestMaxChunkSize:
    def test_max_chunk_size_is_context_minus_overhead(self, mock_book, mock_resume_prompt):
        mock_llm = MagicMock()
        mock_resume_prompt.overhead = 200
        r = BookResumer(mock_book, mock_llm, context_window_size=1000, resume_prompt=mock_resume_prompt)
        assert r.max_chunk_size == 800

    def test_zero_overhead(self, mock_book, mock_resume_prompt):
        mock_llm = MagicMock()
        mock_resume_prompt.overhead = 0
        r = BookResumer(mock_book, mock_llm, context_window_size=500, resume_prompt=mock_resume_prompt)
        assert r.max_chunk_size == 500


class TestBuildSplits:
    def test_content_fits_in_single_chunk(self):
        content = "Short content"
        splits = BookResumer.build_splits(content, 1000)
        assert splits == [(0, len(content))]

    def test_splits_at_newline_after_max_chunk(self):
        content = "A" * 100 + "\n" + "B" * 100
        splits = BookResumer.build_splits(content, 50)
        assert len(splits) == 2
        assert splits[0] == (0, 101)   # 100 A's + newline
        assert splits[1] == (101, 201)

    def test_no_newline_falls_back_to_end_of_content(self):
        content = "A" * 200
        splits = BookResumer.build_splits(content, 50)
        assert splits == [(0, 200)]

    def test_multiple_chunks(self):
        chunk = "A" * 100 + "\n"
        content = chunk * 4
        splits = BookResumer.build_splits(content, 50)
        assert len(splits) == 4
        assert splits[0][0] == 0
        assert splits[-1][1] == len(content)
        for i in range(len(splits) - 1):
            assert splits[i][1] == splits[i + 1][0]

    def test_empty_content(self):
        splits = BookResumer.build_splits("", 100)
        assert splits == []

    def test_content_exactly_max_chunk_size(self):
        content = "A" * 100
        splits = BookResumer.build_splits(content, 100)
        assert splits == [(0, 100)]


class TestPrefix:
    def test_prefix_format(self, resumer):
        assert resumer._prefix == "en/testbook"


class TestProcessSplit:
    """Tests for _process_split: each chunk is summarised to ~10% of its size."""

    def test_uses_10_percent_target_size(self, resumer):
        split = make_split(0, 0, 200)  # chunk of 200 chars
        captured = {}

        def capture_format(**kw):
            captured.update(kw)
            return "prompt"

        resumer.resume_prompt.format.side_effect = capture_format
        resumer._process_split(split, "A" * 1000)

        assert captured["TARGET_SIZE"] == int(200 * 0.10)  # 20

    def test_passes_chunk_text_and_language(self, resumer):
        split = make_split(0, 10, 30)
        captured = {}

        def capture_format(**kw):
            captured.update(kw)
            return "prompt"

        resumer.resume_prompt.format.side_effect = capture_format
        content = "X" * 10 + "Y" * 20 + "Z" * 10
        resumer._process_split(split, content)

        assert captured["TEXT"] == content[10:30]
        assert captured["LANGUAGE"] == "en"

    def test_skip_already_processed_split(self, resumer):
        split = make_split(0, 0, 100, done=True)
        resumer._process_split(split, "A" * 200)
        resumer.llm.generate_with_retry.assert_not_called()

    def test_saves_split_abstract_with_llm_response(self, resumer):
        split = make_split(0, 0, 100)
        resumer.llm.generate_with_retry.return_value = "the abstract"
        resumer._process_split(split, "A" * 200)
        resumer.book.save_split_abstract.assert_called_once_with(split, "the abstract")
