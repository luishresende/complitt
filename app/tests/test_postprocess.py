import os
import pytest

from app.postprocess import (
    adjust_text,
    replace_divs,
    replace_tables,
    extract_id,
    fix_content_pipeline,
    get_md_content,
    merge_md_files,
    get_md_files,
)


class TestExtractId:
    def test_valid_filename(self):
        assert extract_id("page_001.md") == 1

    def test_valid_filename_large_number(self):
        assert extract_id("book_123.md") == 123

    def test_valid_filename_zero_padded(self):
        assert extract_id("page_0042.md") == 42

    def test_invalid_extension_raises(self):
        with pytest.raises(ValueError):
            extract_id("page_001.txt")

    def test_no_id_raises(self):
        with pytest.raises(ValueError):
            extract_id("page.md")

    def test_no_underscore_raises(self):
        with pytest.raises(ValueError):
            extract_id("001.md")


class TestReplaceDivs:
    def test_div_with_alt_uppercased(self):
        html = '<div alt="my image">content</div>'
        assert replace_divs(html) == "[MY IMAGE]"

    def test_div_without_alt_removed(self):
        html = "<div>some content</div>"
        assert replace_divs(html) == ""

    def test_no_div_unchanged(self):
        text = "plain text"
        assert replace_divs(text) == "plain text"

    def test_multiple_divs(self):
        html = '<div alt="fig1">a</div> text <div alt="fig2">b</div>'
        assert replace_divs(html) == "[FIG1] text [FIG2]"

    def test_case_insensitive_alt(self):
        html = '<div ALT="Photo">img</div>'
        assert replace_divs(html) == "[PHOTO]"


class TestReplaceTables:
    def test_simple_table_converted(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        result = replace_tables(html)
        assert "A" in result and "B" in result
        assert "1" in result and "2" in result
        assert "<table" not in result

    def test_no_table_unchanged(self):
        text = "just text"
        assert replace_tables(text) == "just text"


class TestAdjustText:
    def test_ocr_hyphenation_joined(self):
        result = adjust_text("word-\ncontinuation")
        assert result == "wordcontinuation"

    def test_sentence_continuation_joined_with_space(self):
        result = adjust_text("First line\nSecond line")
        assert result == "First line Second line"

    def test_sentence_ending_creates_paragraph_break(self):
        result = adjust_text("First sentence.\nNew paragraph")
        assert "First sentence." in result
        assert "New paragraph" in result
        assert "\n\n" in result

    def test_word_duplication_removed(self):
        # OCR artefact: "Paris\nParis is beautiful" → "Paris is beautiful"
        result = adjust_text("Paris\nParis is beautiful")
        assert result == "Paris is beautiful"

    def test_heading_not_merged(self):
        result = adjust_text("Some text\n## Heading")
        assert "## Heading" in result
        assert "\n\n" in result

    def test_empty_string(self):
        assert adjust_text("") == ""


class TestFixContentPipeline:
    def test_empty_pipeline(self):
        assert fix_content_pipeline("hello") == "hello"

    def test_single_function(self):
        result = fix_content_pipeline("hello", str.upper)
        assert result == "HELLO"

    def test_multiple_functions_applied_in_order(self):
        result = fix_content_pipeline("  hello  ", str.strip, str.upper)
        assert result == "HELLO"


class TestGetMdContent:
    def test_returns_existing_post_processed_file(self, tmp_path):
        (tmp_path / "book_content.md").write_text("already processed")
        assert get_md_content(str(tmp_path)) == "already processed"

    def test_nonexistent_input_raises(self, tmp_path):
        # neither book_content.md nor a paddle_output directory exists
        with pytest.raises(ValueError, match="does not exist"):
            get_md_content(str(tmp_path))

    def test_directory_with_no_md_files_raises(self, tmp_path):
        (tmp_path / "paddle_output").mkdir()
        with pytest.raises(ValueError):
            get_md_content(str(tmp_path))

    def test_merges_md_files_from_paddle_output(self, tmp_path):
        paddle = tmp_path / "paddle_output"
        paddle.mkdir()
        (paddle / "page_001.md").write_text("First page")
        (paddle / "page_002.md").write_text("Second page")
        result = get_md_content(str(tmp_path))
        assert "First page" in result
        assert "Second page" in result

    def test_merged_files_excluded(self, tmp_path):
        paddle = tmp_path / "paddle_output"
        paddle.mkdir()
        (paddle / "page_001.md").write_text("Real content")
        (paddle / "merged_001.md").write_text("Should be ignored")
        result = get_md_content(str(tmp_path))
        assert "Real content" in result
        assert "Should be ignored" not in result


class TestGetMdFiles:
    def test_files_sorted_by_id(self, tmp_path):
        (tmp_path / "page_002.md").write_text("")
        (tmp_path / "page_001.md").write_text("")
        (tmp_path / "page_003.md").write_text("")
        files = get_md_files(str(tmp_path))
        ids = [extract_id(f) for f in files]
        assert ids == sorted(ids)

    def test_merged_files_excluded(self, tmp_path):
        (tmp_path / "page_001.md").write_text("")
        (tmp_path / "merged_001.md").write_text("")
        files = get_md_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "merged_001.md" not in basenames
        assert "page_001.md" in basenames

    def test_non_md_files_excluded(self, tmp_path):
        (tmp_path / "page_001.md").write_text("")
        (tmp_path / "notes.txt").write_text("")
        files = get_md_files(str(tmp_path))
        assert all(f.endswith(".md") for f in files)


class TestMergeMdFiles:
    def test_merges_with_double_newline(self, tmp_path):
        f1 = tmp_path / "page_001.md"
        f2 = tmp_path / "page_002.md"
        f1.write_text("Part one")
        f2.write_text("Part two")
        result = merge_md_files([str(f1), str(f2)])
        assert result == "Part one\n\nPart two"

    def test_empty_files_skipped(self, tmp_path):
        f1 = tmp_path / "page_001.md"
        f2 = tmp_path / "page_002.md"
        f1.write_text("")
        f2.write_text("Content")
        result = merge_md_files([str(f1), str(f2)])
        assert result == "Content"
