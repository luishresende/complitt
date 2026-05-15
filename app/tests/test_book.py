import os
import pytest
from unittest.mock import MagicMock

from app.services.book import Book


class TestBookPaths:
    def test_provider_path(self, tmp_path):
        book = Book("mybook", "en", "openai", [], books_root=str(tmp_path))
        expected = os.path.join(str(tmp_path), "en", "mybook", "providers", "openai")
        assert book.provider_path == expected

    def test_meta_path_ends_with_meta_json(self, tmp_path):
        book = Book("mybook", "en", "openai", [], books_root=str(tmp_path))
        assert book.meta_path.endswith("meta.json")
        assert "mybook" in book.meta_path

    def test_batch_path_by_number(self, tmp_path):
        book = Book("mybook", "en", "openai", [], books_root=str(tmp_path))
        path = book._batch_path(1)
        assert path.endswith("1")
        assert "batches" in path

    def test_splits_path_inside_batch(self, tmp_path):
        book = Book("mybook", "en", "openai", [], books_root=str(tmp_path))
        path = book._splits_path(1)
        assert path.endswith("splits")


class TestBookToDict:
    def test_to_dict_contains_all_fields(self, tmp_path):
        batches = [[{"index": 0, "initial_pos": 0, "end_pos": 100, "content_file": None}]]
        book = Book("mybook", "en", "openai", batches, books_root=str(tmp_path))
        d = book._to_dict()
        assert d == {
            "name": "mybook",
            "language": "en",
            "provider": "openai",
            "batches": batches,
        }


class TestBookCurrentBatch:
    def test_returns_last_batch(self, tmp_path):
        batch1 = [{"index": 0}]
        batch2 = [{"index": 1}]
        book = Book("b", "en", "p", [batch1, batch2], books_root=str(tmp_path))
        assert book.current_batch is batch2

    def test_single_batch(self, tmp_path):
        batch = [{"index": 0}]
        book = Book("b", "en", "p", [batch], books_root=str(tmp_path))
        assert book.current_batch is batch


class TestBookSaveLoad:
    def test_save_creates_meta_json(self, tmp_path):
        book = Book("mybook", "en", "openai", [], books_root=str(tmp_path))
        os.makedirs(os.path.dirname(book.meta_path), exist_ok=True)
        book.save()
        assert os.path.exists(book.meta_path)

    def test_save_and_load_roundtrip(self, tmp_path):
        batches = [[{"index": 0, "initial_pos": 0, "end_pos": 50, "content_file": None}]]
        book = Book("mybook", "en", "openai", batches, books_root=str(tmp_path))
        os.makedirs(os.path.dirname(book.meta_path), exist_ok=True)
        book.save()

        loaded = Book.load("mybook", "en", "openai", books_root=str(tmp_path))
        assert loaded.name == "mybook"
        assert loaded.language == "en"
        assert loaded.provider == "openai"
        assert loaded.batches == batches

    def test_load_nonexistent_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Book.load("missing", "en", "openai", books_root=str(tmp_path))

    def test_reload_updates_batches(self, tmp_path):
        book = Book("mybook", "en", "openai", [], books_root=str(tmp_path))
        os.makedirs(os.path.dirname(book.meta_path), exist_ok=True)
        book.save()

        new_batch = [[{"index": 0, "initial_pos": 0, "end_pos": 10, "content_file": None}]]
        updated = Book("mybook", "en", "openai", new_batch, books_root=str(tmp_path))
        updated.save()

        book.reload()
        assert book.batches == new_batch


class TestBookCreate:
    def test_create_new_book(self, tmp_path):
        splits = [(0, 100), (100, 200)]
        book = Book.create("newbook", "en", splits, "openai", books_root=str(tmp_path))

        assert book.name == "newbook"
        assert len(book.batches) == 1
        assert len(book.batches[0]) == 2
        assert book.batches[0][0] == {"index": 0, "initial_pos": 0, "end_pos": 100, "content_file": None}
        assert book.batches[0][1] == {"index": 1, "initial_pos": 100, "end_pos": 200, "content_file": None}

    def test_create_existing_book_loads_instead(self, tmp_path):
        splits = [(0, 100)]
        book1 = Book.create("mybook", "en", splits, "openai", books_root=str(tmp_path))
        book2 = Book.create("mybook", "en", [(0, 999)], "openai", books_root=str(tmp_path))

        assert book2.batches == book1.batches

    def test_create_makes_splits_directory(self, tmp_path):
        splits = [(0, 100)]
        book = Book.create("newbook", "en", splits, "openai", books_root=str(tmp_path))
        assert os.path.isdir(book._splits_path(1))


class TestBookAddBatch:
    def test_add_batch_increments_batch_count(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        assert len(book.batches) == 1

        book.add_batch([(0, 50), (50, 100)])

        assert len(book.batches) == 2
        assert len(book.batches[1]) == 2

    def test_add_batch_creates_directory(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        book.add_batch([(0, 50)])
        assert os.path.isdir(book._splits_path(2))

    def test_add_batch_persisted_to_disk(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        book.add_batch([(0, 50)])

        reloaded = Book.load("mybook", "en", "openai", books_root=str(tmp_path))
        assert len(reloaded.batches) == 2


class TestSaveSplitAbstract:
    def test_saves_content_and_marks_split(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        split = book.batches[0][0]

        book.save_split_abstract(split, "Abstract text")

        assert split["content_file"] is not None
        assert split["content_file"].endswith(".txt")
        file_path = os.path.join(book._splits_path(1), split["content_file"])
        assert open(file_path).read() == "Abstract text"

    def test_raises_on_empty_string(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        split = book.batches[0][0]

        with pytest.raises(ValueError, match="empty"):
            book.save_split_abstract(split, "")

    def test_raises_on_whitespace_only(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        split = book.batches[0][0]

        with pytest.raises(ValueError, match="empty"):
            book.save_split_abstract(split, "   \n  ")

    def test_raises_on_none(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        split = book.batches[0][0]

        with pytest.raises((ValueError, TypeError)):
            book.save_split_abstract(split, None)

    def test_does_not_mark_split_on_empty(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        split = book.batches[0][0]

        with pytest.raises((ValueError, TypeError)):
            book.save_split_abstract(split, "")

        assert split["content_file"] is None

    def test_does_not_create_file_on_empty(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 100)], "openai", books_root=str(tmp_path))
        split = book.batches[0][0]
        splits_dir = book._splits_path(1)

        with pytest.raises((ValueError, TypeError)):
            book.save_split_abstract(split, "")

        assert not any(os.scandir(splits_dir))


class TestBookLoadAbstracts:
    def test_load_abstracts_reads_split_files(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 50), (50, 100)], "openai", books_root=str(tmp_path))
        splits_dir = book._splits_path(1)

        for i, text in enumerate(["Abstract A", "Abstract B"]):
            filename = f"{i:04d}.txt"
            with open(os.path.join(splits_dir, filename), "w") as f:
                f.write(text)
            book.batches[0][i]["content_file"] = filename

        abstracts = book.load_abstracts()
        assert abstracts == ["Abstract A", "Abstract B"]

    def test_load_abstracts_skips_splits_without_content_file(self, tmp_path):
        book = Book.create("mybook", "en", [(0, 50), (50, 100)], "openai", books_root=str(tmp_path))
        splits_dir = book._splits_path(1)

        with open(os.path.join(splits_dir, "0000.txt"), "w") as f:
            f.write("Only abstract")
        book.batches[0][0]["content_file"] = "0000.txt"

        abstracts = book.load_abstracts()
        assert abstracts == ["Only abstract"]
