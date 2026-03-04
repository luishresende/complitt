import os
import json

from app.llms import LLMClient

BASE_ABSTRACT_PROMPT = None
BASE_SUMMARIZE_PROMPT = None
BOOKS_ROOT = "books"


# ---------- helpers de path ----------

def _book_provider_path(book_name: str, language: str, provider: str) -> str:
    return os.path.join(BOOKS_ROOT, language, book_name, "providers", provider)

def _splits_path(book_name: str, language: str, provider: str) -> str:
    return os.path.join(_book_provider_path(book_name, language, provider), "splits")

def _meta_path(book_name: str, language: str, provider: str) -> str:
    return os.path.join(_book_provider_path(book_name, language, provider), "meta.json")


# ---------- criação do book ----------

def create_book(book_name: str, language: str, splits: list[tuple[int, int]], provider: str) -> dict:
    """
    Cria a estrutura de pastas e persiste os metadados do book (splits).
    Retorna o objeto book como dicionário.
    """
    splits_dir = _splits_path(book_name, language, provider)
    os.makedirs(splits_dir, exist_ok=True)

    meta_path = _meta_path(book_name, language, provider)
    if os.path.exists(meta_path):
        with open((meta_path), "r") as f:
            data = json.load(f)
            return data

    book = {
        "name": book_name,
        "language": language,
        "provider": provider,
        "splits": [
            {"index": i, "initial_pos": s[0], "end_pos": s[1], "content_file": None}
            for i, s in enumerate(splits)
        ]
    }

    _save_meta(book_name, language, provider, book)
    return book


def _save_meta(book_name: str, language: str, provider: str, book: dict):
    with open(_meta_path(book_name, language, provider), "w") as f:
        json.dump(book, f, indent=2)


# ---------- leitura ----------

def get_book(book_name: str, language: str, provider: str) -> dict:
    meta = _meta_path(book_name, language, provider)
    if not os.path.exists(meta):
        raise FileNotFoundError(f"Book not found: {meta}")
    with open(meta, "r") as f:
        return json.load(f)


# ---------- geração de abstracts ----------

def generate_abstract(text: str, book: dict, split: dict, llm: LLMClient) -> dict:
    global BASE_ABSTRACT_PROMPT

    if not BASE_ABSTRACT_PROMPT:
        with open("app/prompts/resume_prompt.txt", "r") as f:
            BASE_ABSTRACT_PROMPT = f.read()

    chunk = text[split["initial_pos"]:split["end_pos"]]
    target_size = int(len(chunk) * 0.10)
    prompt = BASE_ABSTRACT_PROMPT.replace("{TARGET_SIZE}", str(target_size)).replace("{TEXT}", chunk)

    response = llm.generate_content(prompt)

    provider = book["provider"]
    filename = f"{split['index']:04d}.txt"
    filepath = os.path.join(_splits_path(book["name"], book["language"], provider), filename)
    with open(filepath, "w") as f:
        f.write(response)

    split["content_file"] = filename
    _save_meta(book["name"], book["language"], provider, book)

    return split


# ---------- summarização ----------

def summarize_abstracts(abstracts: list[str], llm: LLMClient) -> str:
    global BASE_SUMMARIZE_PROMPT
    if not BASE_SUMMARIZE_PROMPT:
        with open("app/prompts/summarize_prompt.txt", "r") as f:
            BASE_SUMMARIZE_PROMPT = f.read()

    abstracts_text = "\n\n".join(abstracts)
    prompt = BASE_SUMMARIZE_PROMPT.replace("{ABSTRACT_CONTENTS}", abstracts_text)
    response = llm.generate_content(prompt)
    return response


# ---------- leitura de abstracts ----------

def load_abstracts(book: dict) -> list[str]:
    """Lê todos os abstracts já gerados do disco, em ordem."""
    abstracts = []
    provider = book["provider"]
    splits_dir = _splits_path(book["name"], book["language"], provider)
    for split in sorted(book["splits"], key=lambda s: s["index"]):
        if split["content_file"]:
            with open(os.path.join(splits_dir, split["content_file"]), "r") as f:
                abstracts.append(f.read())
    return abstracts
