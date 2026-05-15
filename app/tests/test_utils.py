import pytest

from app.utils import validate_input_dir_structure, prepare_dir


class TestValidateInputDirStructure:
    def test_valid_structure(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        book = lang / "mybook"
        book.mkdir()
        (book / "mybook.pdf").write_bytes(b"%PDF")

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert valid
        assert issues == []

    def test_pdf_outside_book_folder(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        (lang / "mybook.pdf").write_bytes(b"%PDF")

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert not valid
        assert any("PDF outside book folder" in issue for issue in issues)

    def test_no_pdf_in_book_folder(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        (lang / "mybook").mkdir()

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert not valid
        assert any("No PDF found" in issue for issue in issues)

    def test_multiple_pdfs_in_book_folder(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        book = lang / "mybook"
        book.mkdir()
        (book / "mybook.pdf").write_bytes(b"%PDF")
        (book / "extra.pdf").write_bytes(b"%PDF")

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert not valid
        assert any("Multiple PDFs" in issue for issue in issues)

    def test_pdf_name_does_not_match_folder(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        book = lang / "mybook"
        book.mkdir()
        (book / "wrongname.pdf").write_bytes(b"%PDF")

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert not valid
        assert any("does not match" in issue for issue in issues)

    def test_multiple_languages(self, tmp_path):
        for lang_name in ("en", "pt"):
            lang = tmp_path / lang_name
            lang.mkdir()
            book = lang / "book"
            book.mkdir()
            (book / "book.pdf").write_bytes(b"%PDF")

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert valid
        assert issues == []

    def test_non_directory_items_at_root_ignored(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert valid
        assert issues == []

    def test_multiple_books_one_invalid(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()

        good_book = lang / "goodbook"
        good_book.mkdir()
        (good_book / "goodbook.pdf").write_bytes(b"%PDF")

        bad_book = lang / "badbook"
        bad_book.mkdir()
        # no PDF inside

        valid, issues = validate_input_dir_structure(str(tmp_path))

        assert not valid
        assert len(issues) == 1
        assert "badbook" in issues[0]


class TestPrepareDir:
    def test_pdf_moved_into_book_folder(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        (lang / "mybook.pdf").write_bytes(b"%PDF")

        prepare_dir(str(tmp_path))

        assert (lang / "mybook" / "mybook.pdf").exists()
        assert not (lang / "mybook.pdf").exists()

    def test_pdf_already_in_correct_folder_untouched(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        book = lang / "mybook"
        book.mkdir()
        (book / "mybook.pdf").write_bytes(b"%PDF")

        prepare_dir(str(tmp_path))

        assert (book / "mybook.pdf").exists()

    def test_multiple_pdfs_in_language_folder(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        (lang / "book1.pdf").write_bytes(b"%PDF")
        (lang / "book2.pdf").write_bytes(b"%PDF")

        prepare_dir(str(tmp_path))

        assert (lang / "book1" / "book1.pdf").exists()
        assert (lang / "book2" / "book2.pdf").exists()

    def test_non_pdf_files_ignored(self, tmp_path):
        lang = tmp_path / "en"
        lang.mkdir()
        (lang / "notes.txt").write_text("ignore me")

        prepare_dir(str(tmp_path))

        assert (lang / "notes.txt").exists()
        assert not (lang / "notes").exists()
