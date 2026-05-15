import os
import json
import threading


_write_lock = threading.Lock()


class Book:
    def __init__(self, name: str, language: str, provider: str, batches: list, books_root: str = "books"):
        self.name = name
        self.language = language
        self.provider = provider
        self.batches = batches
        self.books_root = books_root

    # --- path helpers ---

    @property
    def book_path(self) -> str:
        return os.path.join(self.books_root, self.language, self.name)

    @property
    def provider_path(self) -> str:
        return os.path.join(self.books_root, self.language, self.name, "providers", self.provider)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.provider_path, "meta.json")

    def _batch_path(self, batch_num: int) -> str:
        batch_root = os.path.join(self.provider_path, "batches")
        if batch_num == -1:
            entries = os.listdir(batch_root)
            return os.path.join(batch_root, str(max(entries, key=lambda x: int(x))))
        return os.path.join(batch_root, str(batch_num))

    def _splits_path(self, batch_num: int) -> str:
        return os.path.join(self._batch_path(batch_num), "splits")

    # --- persistence ---

    def save(self):
        with _write_lock:
            with open(self.meta_path, "w") as f:
                json.dump(self._to_dict(), f, indent=2)

    def reload(self):
        with open(self.meta_path, "r") as f:
            data = json.load(f)
        self.batches = data["batches"]

    def _to_dict(self) -> dict:
        return {
            "name": self.name,
            "language": self.language,
            "provider": self.provider,
            "batches": self.batches,
        }

    # --- factory methods ---

    @classmethod
    def create(cls, name: str, language: str, splits: list[tuple[int, int]], provider: str, books_root: str = "books") -> "Book":
        book = cls(name, language, provider, batches=[], books_root=books_root)

        os.makedirs(book._splits_path(1), exist_ok=True)

        if os.path.exists(book.meta_path):
            return cls.load(name, language, provider, books_root)

        book.batches = [[
            {"index": i, "initial_pos": s[0], "end_pos": s[1], "content_file": None}
            for i, s in enumerate(splits)
        ]]
        book.save()
        return book

    @classmethod
    def load(cls, name: str, language: str, provider: str, books_root: str = "books") -> "Book":
        meta = os.path.join(books_root, language, name, "providers", provider, "meta.json")
        if not os.path.exists(meta):
            raise FileNotFoundError(f"Book not found: {meta}")
        with open(meta, "r") as f:
            data = json.load(f)
        return cls(data["name"], data["language"], data["provider"], data["batches"], books_root)

    # --- batch management ---

    @property
    def current_batch(self) -> list:
        return self.batches[-1]

    def add_batch(self, splits: list[tuple[int, int]]):
        batch_num = len(self.batches) + 1
        os.makedirs(self._splits_path(batch_num), exist_ok=True)
        self.batches.append([
            {"index": i, "initial_pos": s[0], "end_pos": s[1], "content_file": None}
            for i, s in enumerate(splits)
        ])
        self.save()

    # --- split persistence ---

    def save_split_abstract(self, split: dict, content: str):
        if not content or not content.strip():
            raise ValueError(f"Cannot save empty content for split {split['index']}")
        batch_num = len(self.batches)
        filename = f"{split['index']:04d}.txt"
        with open(os.path.join(self._splits_path(batch_num), filename), "w") as f:
            f.write(content)
        split["content_file"] = filename
        self.save()

    def load_abstracts(self) -> list[str]:
        abstracts = []
        splits_dir = self._splits_path(-1)
        for split in sorted(self.current_batch, key=lambda s: s["index"]):
            if split["content_file"]:
                with open(os.path.join(splits_dir, split["content_file"]), "r") as f:
                    abstracts.append(f.read())
        return abstracts
