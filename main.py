import os
from app.preprocess import fix_content_pipeline, adjust_text, replace_divs, get_md_files, merge_md_files, replace_tables
from app.services.book import (
    create_book, get_book,
    generate_abstract, summarize_abstracts, load_abstracts
)
from app.llms import LLMClient
from app.llms.gemini_client import GeminiClient
from app.llms.hugging_face_client import HuggingFaceLLMClient

CHAR_PER_PAGE = 2000
MINIMUM_CHARS_PER_PAGE = CHAR_PER_PAGE * 2
BOOKS_INPUT_ROOT = "../documentos/paddleocr_transcriptions"
BOOKS_OUTPUT_ROOT = "books"


def get_md_content(input_path):
    if not os.path.exists(input_path):
        raise ValueError("Input does not exist")

    if os.path.isfile(input_path) and os.path.splitext(input_path)[1] == ".md":
        with open(input_path, "r") as f:
            md_content = f.read()
        md_content = fix_content_pipeline(md_content, replace_divs, replace_tables)

    elif os.path.isdir(input_path):
        md_files = get_md_files(input_path)
        if not md_files:
            raise ValueError("Input is not a markdown file or a path with markdown files")

        md_content = merge_md_files(md_files)
        with open(os.path.join(input_path, 'merged_debug.md'), "w") as f:
            f.write(md_content)
        md_content = fix_content_pipeline(md_content, replace_divs, adjust_text, replace_tables)

    return md_content


def build_splits(content: str) -> list[tuple[int, int]]:
    initial_index = 0
    splits = []
    while initial_index < len(content):
        search_start = initial_index + MINIMUM_CHARS_PER_PAGE
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


def process_book(book_name: str, language: str, book_input_path: str, llm: LLMClient):
    book_dir = os.path.join(BOOKS_OUTPUT_ROOT, language, llm.name, book_name)
    meta_path = os.path.join(book_dir, "meta.json")

    if os.path.exists(meta_path):
        print(f"[SKIP] {language}/{book_name} já existe para o provider '{llm.name}'.")
        return

    print(f"[PROCESS] {language}/{book_name}")
    content = get_md_content(book_input_path)
    splits = build_splits(content)

    book = create_book(book_name, language, splits, llm.name)
    for split in book["splits"]:
        print(f"  Generating abstract for split {split['index']}: ({split['initial_pos']}, {split['end_pos']})")
        generate_abstract(content, book, split, llm)

    book = get_book(book_name, language, llm.name)
    abstracts = load_abstracts(book)

    os.makedirs(book_dir, exist_ok=True)
    with open(os.path.join(book_dir, "abstract_contents.md"), "w") as f:
        f.write("\n\n".join(abstracts))

    with open(os.path.join(book_dir, "summary.md"), "w") as f:
        f.write(summarize_abstracts(abstracts, llm))

    with open(os.path.join(book_dir, "summary_without_intro.md"), "w") as f:
        f.write(summarize_abstracts(abstracts[1:], llm))

    print(f"  [DONE] {language}/{book_name}")


llm_provider = GeminiClient("gemini-2.5-flash")

for language in os.listdir(BOOKS_INPUT_ROOT):
    language_path = os.path.join(BOOKS_INPUT_ROOT, language)
    if not os.path.isdir(language_path):
        continue

    for book_name in os.listdir(language_path):
        book_input_path = os.path.join(language_path, book_name)
        if not os.path.isdir(book_input_path):
            continue

        try:
            process_book(book_name, language, book_input_path, llm_provider)
        except Exception as e:
            print(f"[ERROR] {language}/{book_name}: {e}")