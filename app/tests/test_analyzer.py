import pytest
from unittest.mock import MagicMock, call

from app.services.analyzer import BookAnalyzer


@pytest.fixture
def mock_book(tmp_path):
    book = MagicMock()
    book.language = "en"
    book.name = "testbook"
    book.provider_path = str(tmp_path)
    return book


@pytest.fixture
def mock_summarize_prompt():
    prompt = MagicMock()
    prompt.format.side_effect = lambda **kw: f"summarize:{kw['ABSTRACT_CONTENTS']}"
    return prompt


@pytest.fixture
def mock_keywords_prompt():
    prompt = MagicMock()
    prompt.format.side_effect = lambda **kw: f"keywords:{kw['ABSTRACT_CONTENTS']}"
    return prompt


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate_with_retry.return_value = "llm response"
    return llm


@pytest.fixture
def analyzer(mock_book, mock_summarize_prompt, mock_keywords_prompt, mock_llm):
    return BookAnalyzer(mock_book, mock_llm, mock_summarize_prompt, mock_keywords_prompt)


class TestSummarize:
    def test_joins_abstracts_and_calls_llm(self, analyzer, mock_summarize_prompt, mock_llm):
        abstracts = ["part A", "part B"]
        result = analyzer.summarize(abstracts)
        mock_summarize_prompt.format.assert_called_once_with(ABSTRACT_CONTENTS="part A\n\npart B", LANGUAGE="en")
        mock_llm.generate_with_retry.assert_called_once()
        assert result == "llm response"

    def test_returns_llm_output(self, analyzer, mock_llm):
        mock_llm.generate_with_retry.return_value = "the summary"
        assert analyzer.summarize(["x"]) == "the summary"


class TestKeywords:
    def test_joins_abstracts_and_calls_llm(self, analyzer, mock_keywords_prompt, mock_llm):
        abstracts = ["part A", "part B"]
        result = analyzer.keywords(abstracts)
        mock_keywords_prompt.format.assert_called_once_with(ABSTRACT_CONTENTS="part A\n\npart B", LANGUAGE="en")
        mock_llm.generate_with_retry.assert_called_once()
        assert result == "llm response"

    def test_returns_llm_output(self, analyzer, mock_llm):
        mock_llm.generate_with_retry.return_value = "keyword1\nkeyword2"
        assert analyzer.keywords(["x"]) == "keyword1\nkeyword2"


class TestWriteAbstracts:
    def test_writes_abstract_contents_file(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        analyzer.write_abstracts(["part A", "part B"])
        assert (tmp_path / "abstract_contents.txt").read_text() == "part A\n\npart B"

    def test_skips_if_already_exists_and_non_empty(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "abstract_contents.txt").write_text("old content")
        analyzer.write_abstracts(["new content"])
        assert (tmp_path / "abstract_contents.txt").read_text() == "old content"

    def test_overwrites_empty_existing_file(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "abstract_contents.txt").write_text("")
        analyzer.write_abstracts(["new content"])
        assert (tmp_path / "abstract_contents.txt").read_text() == "new content"

    def test_raises_on_empty_abstracts_list(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        with pytest.raises(ValueError, match="No abstracts"):
            analyzer.write_abstracts([])

    def test_does_not_create_file_on_empty_list(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        with pytest.raises(ValueError):
            analyzer.write_abstracts([])
        assert not (tmp_path / "abstract_contents.txt").exists()

    def test_does_not_call_llm(self, tmp_path, analyzer, mock_llm):
        analyzer.book.provider_path = str(tmp_path)
        analyzer.write_abstracts(["x"])
        mock_llm.generate_with_retry.assert_not_called()


class TestWriteSummary:
    def test_writes_summary_and_summary_without_intro(self, tmp_path, analyzer, mock_llm):
        analyzer.book.provider_path = str(tmp_path)
        mock_llm.generate_with_retry.return_value = "generated"
        analyzer.write_summary(["intro", "chapter one"])
        assert (tmp_path / "summary.txt").exists()
        assert (tmp_path / "summary_without_intro.txt").exists()

    def test_summary_without_intro_skips_first_abstract(self, tmp_path, analyzer, mock_llm):
        analyzer.book.provider_path = str(tmp_path)
        call_args = []

        def capture(prompt_text):
            call_args.append(prompt_text)
            return "result"

        mock_llm.generate_with_retry.side_effect = capture
        analyzer.write_summary(["intro", "chapter one", "chapter two"])

        assert "intro" in call_args[0]
        assert "intro" not in call_args[1]

    def test_skips_existing_non_empty_files(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "summary.txt").write_text("old summary")
        (tmp_path / "summary_without_intro.txt").write_text("old summary_without_intro")
        analyzer.write_summary(["new content"])
        assert (tmp_path / "summary.txt").read_text() == "old summary"
        assert (tmp_path / "summary_without_intro.txt").read_text() == "old summary_without_intro"

    def test_overwrites_empty_summary_file(self, tmp_path, analyzer, mock_llm):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "summary.txt").write_text("")
        mock_llm.generate_with_retry.return_value = "fresh summary"
        analyzer.write_summary(["intro", "chapter"])
        assert (tmp_path / "summary.txt").read_text() == "fresh summary"


class TestWriteKeywords:
    def test_writes_keywords_file(self, tmp_path, analyzer, mock_llm):
        analyzer.book.provider_path = str(tmp_path)
        mock_llm.generate_with_retry.return_value = "kw1\nkw2"
        analyzer.write_keywords(["abstract one"])
        assert (tmp_path / "keywords.txt").read_text() == "kw1\nkw2"

    def test_skips_if_already_exists_and_non_empty(self, tmp_path, analyzer, mock_llm):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "keywords.txt").write_text("old keywords")
        analyzer.write_keywords(["new content"])
        assert (tmp_path / "keywords.txt").read_text() == "old keywords"
        mock_llm.generate_with_retry.assert_not_called()

    def test_overwrites_empty_keywords_file(self, tmp_path, analyzer, mock_llm):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "keywords.txt").write_text("")
        mock_llm.generate_with_retry.return_value = "kw1\nkw2"
        analyzer.write_keywords(["abstract"])
        assert (tmp_path / "keywords.txt").read_text() == "kw1\nkw2"


class TestAbstractsExist:
    def test_true_when_file_present_and_non_empty(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "abstract_contents.txt").write_text("content")
        assert analyzer.abstracts_exist() is True

    def test_false_when_file_missing(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        assert analyzer.abstracts_exist() is False

    def test_false_when_file_is_empty(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "abstract_contents.txt").write_text("")
        assert analyzer.abstracts_exist() is False


class TestSummaryExists:
    def test_true_when_both_files_present_and_non_empty(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "summary.txt").write_text("s")
        (tmp_path / "summary_without_intro.txt").write_text("s")
        assert analyzer.summary_exists() is True

    def test_false_when_one_file_missing(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "summary.txt").write_text("s")
        assert analyzer.summary_exists() is False

    def test_false_when_both_missing(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        assert analyzer.summary_exists() is False

    def test_false_when_one_file_is_empty(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "summary.txt").write_text("s")
        (tmp_path / "summary_without_intro.txt").write_text("")
        assert analyzer.summary_exists() is False


class TestKeywordsExist:
    def test_true_when_file_present_and_non_empty(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "keywords.txt").write_text("kw")
        assert analyzer.keywords_exist() is True

    def test_false_when_file_missing(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        assert analyzer.keywords_exist() is False

    def test_false_when_file_is_empty(self, tmp_path, analyzer):
        analyzer.book.provider_path = str(tmp_path)
        (tmp_path / "keywords.txt").write_text("")
        assert analyzer.keywords_exist() is False
