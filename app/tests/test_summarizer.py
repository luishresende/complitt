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


class TestFinalBatchTargetTotal:
    def test_final_batch_target_total_formula(self, mock_book, mock_resume_prompt):
        mock_llm = MagicMock()
        mock_resume_prompt.overhead = 100
        r = BookResumer(mock_book, mock_llm, context_window_size=1000, resume_prompt=mock_resume_prompt)
        # int(1000 * 0.9) - 100 = 900 - 100 = 800
        assert r._final_batch_target_total == 800

    def test_final_batch_target_total_with_large_overhead(self, mock_book, mock_resume_prompt):
        mock_llm = MagicMock()
        mock_resume_prompt.overhead = 400
        r = BookResumer(mock_book, mock_llm, context_window_size=1000, resume_prompt=mock_resume_prompt)
        assert r._final_batch_target_total == 500  # 900 - 400


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


class TestProcessSplitTargetSize:
    """Tests for the target_size selection logic in _process_split."""

    def _make_resumer(self, context_window_size=1000, overhead=100):
        book = MagicMock()
        book.language = "en"
        book.name = "test"
        prompt = MagicMock()
        prompt.overhead = overhead
        prompt.format.side_effect = lambda **kw: f"p:{kw['TEXT']}"
        llm = MagicMock()
        llm.generate_with_retry.return_value = "result"
        return BookResumer(book, llm, context_window_size=context_window_size, resume_prompt=prompt)

    def test_more_than_10_splits_uses_10_percent(self):
        r = self._make_resumer(context_window_size=1000, overhead=100)
        content = "A" * 1000
        split = make_split(0, 0, 200)  # chunk of 200 chars
        captured = {}

        def capture_format(**kw):
            captured.update(kw)
            return "prompt"

        r.resume_prompt.format.side_effect = capture_format
        r._process_split(split, content, num_splits=11)

        assert captured["TARGET_SIZE"] == int(200 * 0.10)  # 20

    def test_exactly_10_splits_uses_proportional(self):
        r = self._make_resumer(context_window_size=1000, overhead=100)
        content = "A" * 1000
        split = make_split(0, 0, 500)  # chunk is 50% of content
        captured = {}

        def capture_format(**kw):
            captured.update(kw)
            return "prompt"

        r.resume_prompt.format.side_effect = capture_format
        r._process_split(split, content, num_splits=10)

        # target_total = int(1000*0.9) - 100 = 800
        # proportional = int(800 * 500/1000) = 400
        assert captured["TARGET_SIZE"] == 400

    def test_fewer_than_10_splits_uses_proportional(self):
        r = self._make_resumer(context_window_size=1000, overhead=100)
        content = "A" * 1000
        split = make_split(0, 0, 250)  # chunk is 25% of content
        captured = {}

        def capture_format(**kw):
            captured.update(kw)
            return "prompt"

        r.resume_prompt.format.side_effect = capture_format
        r._process_split(split, content, num_splits=4)

        # target_total = 800; proportional = int(800 * 250/1000) = 200
        assert captured["TARGET_SIZE"] == 200

    def test_proportional_target_sizes_sum_to_target_total(self):
        """Sum of all proportional target_sizes should approximate _final_batch_target_total."""
        r = self._make_resumer(context_window_size=1000, overhead=100)
        # 4 equal chunks of 250 chars
        content = "A" * 1000
        chunks = [make_split(i, i * 250, (i + 1) * 250) for i in range(4)]
        total = r._final_batch_target_total  # 800
        captured_sizes = []

        def capture_format(**kw):
            captured_sizes.append(kw["TARGET_SIZE"])
            return "prompt"

        r.resume_prompt.format.side_effect = capture_format
        for s in chunks:
            r._process_split(s, content, num_splits=4)

        assert sum(captured_sizes) == total

    def test_skip_already_processed_split(self):
        r = self._make_resumer()
        split = make_split(0, 0, 100, done=True)
        r._process_split(split, "A" * 200, num_splits=5)
        r.llm.generate_with_retry.assert_not_called()

    def test_target_size_never_zero(self):
        """Very small chunks with large overhead should yield target_size >= 1."""
        r = self._make_resumer(context_window_size=100, overhead=95)
        # target_total = int(100*0.9) - 95 = 90 - 95 = -5 → clamped to 1 via max(1, ...)
        content = "A" * 100
        split = make_split(0, 0, 1)
        captured = {}

        def capture_format(**kw):
            captured.update(kw)
            return "prompt"

        r.resume_prompt.format.side_effect = capture_format
        r._process_split(split, content, num_splits=1)
        assert captured["TARGET_SIZE"] >= 1
