import os
import pytest
from unittest.mock import MagicMock, patch, call

import app.summarize as summarize_module
from app.summarize import (
    process_book_keywords,
    process_book_abstract,
    _iter_books,
    _build_llm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(name="mock-model"):
    llm = MagicMock()
    llm.name = name
    return llm


def _make_analyzer(*, abstracts_exist=True, keywords_exist=False, summary_exists=False):
    analyzer = MagicMock()
    analyzer.abstracts_exist.return_value = abstracts_exist
    analyzer.keywords_exist.return_value = keywords_exist
    analyzer.summary_exists.return_value = summary_exists
    return analyzer


# ---------------------------------------------------------------------------
# _build_llm
# ---------------------------------------------------------------------------

class TestBuildLlm:
    def test_raises_for_unknown_api(self):
        with pytest.raises(ValueError, match="not supported"):
            _build_llm("unknown_api", "some-model")

    @pytest.mark.parametrize("api", ["openai", "anthropic", "gemini", "xai", "mistral", "deepseek"])
    def test_instantiates_known_providers(self, api):
        fake_cls = MagicMock(return_value=MagicMock())
        with patch.dict(summarize_module._LLM_PROVIDERS, {api: fake_cls}):
            _build_llm(api, "my-model")
            fake_cls.assert_called_once_with("my-model")


# ---------------------------------------------------------------------------
# _iter_books
# ---------------------------------------------------------------------------

class TestIterBooks:
    def test_yields_language_and_book_name(self, tmp_path):
        (tmp_path / "EN" / "book_a").mkdir(parents=True)
        (tmp_path / "EN" / "book_b").mkdir(parents=True)
        results = set(_iter_books(str(tmp_path)))
        assert ("EN", "book_a") in results
        assert ("EN", "book_b") in results

    def test_skips_files_at_language_level(self, tmp_path):
        (tmp_path / "EN").mkdir()
        (tmp_path / "EN" / "book_a").mkdir()
        (tmp_path / "EN" / "not_a_book.txt").write_text("file")
        results = list(_iter_books(str(tmp_path)))
        assert all(book == "book_a" for _, book in results)

    def test_skips_non_directory_language_entries(self, tmp_path):
        (tmp_path / "EN").mkdir()
        (tmp_path / "EN" / "book_a").mkdir()
        (tmp_path / "README.md").write_text("file")
        results = list(_iter_books(str(tmp_path)))
        assert all(lang == "EN" for lang, _ in results)

    def test_empty_root_yields_nothing(self, tmp_path):
        assert list(_iter_books(str(tmp_path))) == []


# ---------------------------------------------------------------------------
# process_book_keywords
# ---------------------------------------------------------------------------

class TestProcessBookKeywords:
    def _prompts(self):
        return MagicMock(), MagicMock()

    def test_skips_when_book_not_found(self, caplog):
        llm = _make_llm()
        sp, kp = self._prompts()
        with patch("app.summarize.Book.load", side_effect=FileNotFoundError):
            process_book_keywords("missing_book", "EN", llm, sp, kp)
        assert "summarize" in caplog.text.lower() or "skip" in caplog.text.lower()

    def test_skips_when_abstracts_missing(self, caplog):
        llm = _make_llm()
        sp, kp = self._prompts()
        mock_book = MagicMock()
        analyzer = _make_analyzer(abstracts_exist=False)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer):
            process_book_keywords("book", "EN", llm, sp, kp)
        analyzer.write_keywords.assert_not_called()

    def test_skips_when_keywords_already_exist(self):
        llm = _make_llm()
        sp, kp = self._prompts()
        mock_book = MagicMock()
        analyzer = _make_analyzer(abstracts_exist=True, keywords_exist=True)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer):
            process_book_keywords("book", "EN", llm, sp, kp)
        analyzer.write_keywords.assert_not_called()

    def test_calls_write_keywords_with_loaded_abstracts(self):
        llm = _make_llm()
        sp, kp = self._prompts()
        mock_book = MagicMock()
        mock_book.load_abstracts.return_value = ["abstract one", "abstract two"]
        analyzer = _make_analyzer(abstracts_exist=True, keywords_exist=False)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer):
            process_book_keywords("book", "EN", llm, sp, kp)
        analyzer.write_keywords.assert_called_once_with(["abstract one", "abstract two"])

    def test_passes_correct_prompts_to_analyzer(self):
        llm = _make_llm()
        sp, kp = MagicMock(), MagicMock()
        mock_book = MagicMock()
        mock_book.load_abstracts.return_value = []
        analyzer = _make_analyzer(abstracts_exist=True, keywords_exist=False)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer) as MockAnalyzer:
            process_book_keywords("book", "EN", llm, sp, kp)
        MockAnalyzer.assert_called_once_with(mock_book, llm, sp, kp)


# ---------------------------------------------------------------------------
# process_book_final_summary
# ---------------------------------------------------------------------------

class TestProcessBookAbstract:
    def _prompts(self):
        return MagicMock(), MagicMock()

    def test_skips_when_book_not_found(self, caplog):
        llm = _make_llm()
        ap, kp = self._prompts()
        with patch("app.summarize.Book.load", side_effect=FileNotFoundError):
            process_book_abstract("missing_book", "EN", llm, ap, kp)
        assert "resume" in caplog.text.lower() or "skip" in caplog.text.lower()

    def test_skips_when_abstracts_missing(self):
        llm = _make_llm()
        ap, kp = self._prompts()
        mock_book = MagicMock()
        analyzer = _make_analyzer(abstracts_exist=False)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer):
            process_book_abstract("book", "EN", llm, ap, kp)
        analyzer.write_summary.assert_not_called()

    def test_skips_when_abstract_already_exists(self):
        llm = _make_llm()
        ap, kp = self._prompts()
        mock_book = MagicMock()
        analyzer = _make_analyzer(abstracts_exist=True, summary_exists=True)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer):
            process_book_abstract("book", "EN", llm, ap, kp)
        analyzer.write_summary.assert_not_called()

    def test_calls_write_summary_with_loaded_abstracts(self):
        llm = _make_llm()
        ap, kp = self._prompts()
        mock_book = MagicMock()
        mock_book.load_abstracts.return_value = ["intro", "chapter one"]
        analyzer = _make_analyzer(abstracts_exist=True, summary_exists=False)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer):
            process_book_abstract("book", "EN", llm, ap, kp)
        analyzer.write_summary.assert_called_once_with(["intro", "chapter one"])

    def test_passes_correct_prompts_to_analyzer(self):
        llm = _make_llm()
        ap, kp = MagicMock(), MagicMock()
        mock_book = MagicMock()
        mock_book.load_abstracts.return_value = []
        analyzer = _make_analyzer(abstracts_exist=True, summary_exists=False)
        with patch("app.summarize.Book.load", return_value=mock_book), \
             patch("app.summarize.BookAnalyzer", return_value=analyzer) as MockAnalyzer:
            process_book_abstract("book", "EN", llm, ap, kp)
        MockAnalyzer.assert_called_once_with(mock_book, llm, ap, kp)


# ---------------------------------------------------------------------------
# Top-level: keywords() and final_summary()
# ---------------------------------------------------------------------------

class TestKeywordsTopLevel:
    def test_iterates_all_books(self, tmp_path):
        (tmp_path / "EN" / "book_a").mkdir(parents=True)
        (tmp_path / "EN" / "book_b").mkdir(parents=True)
        called_with = []

        def fake_process(book_name, language, llm, sp, kp):
            called_with.append((book_name, language))

        with patch.dict(summarize_module._LLM_PROVIDERS, {"mock_api": MagicMock(return_value=_make_llm())}), \
             patch("app.summarize.process_book_keywords", side_effect=fake_process), \
             patch("app.summarize._load_prompts", return_value=(MagicMock(), MagicMock(), MagicMock())):
            summarize_module.keywords(str(tmp_path), "mock_api", "model")

        assert set(called_with) == {("book_a", "EN"), ("book_b", "EN")}

    def test_continues_after_error(self, tmp_path):
        (tmp_path / "EN" / "book_a").mkdir(parents=True)
        (tmp_path / "EN" / "book_b").mkdir(parents=True)
        called_with = []

        def fake_process(book_name, language, llm, sp, kp):
            if book_name == "book_a":
                raise RuntimeError("oops")
            called_with.append(book_name)

        with patch.dict(summarize_module._LLM_PROVIDERS, {"mock_api": MagicMock(return_value=_make_llm())}), \
             patch("app.summarize.process_book_keywords", side_effect=fake_process), \
             patch("app.summarize._load_prompts", return_value=(MagicMock(), MagicMock(), MagicMock())):
            summarize_module.keywords(str(tmp_path), "mock_api", "model")

        assert "book_b" in called_with


class TestAbstractTopLevel:
    def test_iterates_all_books(self, tmp_path):
        (tmp_path / "PT" / "book_x").mkdir(parents=True)
        called_with = []

        def fake_process(book_name, language, llm, ap, kp):
            called_with.append((book_name, language))

        with patch.dict(summarize_module._LLM_PROVIDERS, {"mock_api": MagicMock(return_value=_make_llm())}), \
             patch("app.summarize.process_book_abstract", side_effect=fake_process), \
             patch("app.summarize._load_prompts", return_value=(MagicMock(), MagicMock(), MagicMock())):
            summarize_module.abstract(str(tmp_path), "mock_api", "model")

        assert called_with == [("book_x", "PT")]

    def test_continues_after_error(self, tmp_path):
        (tmp_path / "EN" / "book_a").mkdir(parents=True)
        (tmp_path / "EN" / "book_b").mkdir(parents=True)
        called_with = []

        def fake_process(book_name, language, llm, ap, kp):
            if book_name == "book_a":
                raise RuntimeError("oops")
            called_with.append(book_name)

        with patch.dict(summarize_module._LLM_PROVIDERS, {"mock_api": MagicMock(return_value=_make_llm())}), \
             patch("app.summarize.process_book_abstract", side_effect=fake_process), \
             patch("app.summarize._load_prompts", return_value=(MagicMock(), MagicMock(), MagicMock())):
            summarize_module.abstract(str(tmp_path), "mock_api", "model")

        assert "book_b" in called_with
