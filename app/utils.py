import shutil

def prepare_dir(books_path):
    """
    Organize PDFs into the expected structure:

        books_path/
            <language>/
                <book_name>/
                    book_name.pdf

    Each language directory may contain PDFs directly.
    A folder is created for each PDF and the file is moved into it.

    Args:
        books_path (str): Root directory containing language folders.
    """

    found_books = []

    for language in os.listdir(books_path):
        language_path = os.path.join(books_path, language)

        if not os.path.isdir(language_path):
            continue

        for root, dirs, files in os.walk(language_path):
            for file in files:
                if not file.lower().endswith(".pdf"):
                    continue

                pdf_name = os.path.splitext(file)[0]
                pdf_path = os.path.join(root, file)

                # already inside correct book folder
                if os.path.basename(root) == pdf_name:
                    continue

                folder_path = os.path.join(root, pdf_name)
                os.makedirs(folder_path, exist_ok=True)

                dest_path = os.path.join(folder_path, file)
                shutil.move(pdf_path, dest_path)

                found_books.append(f"{language}/{pdf_name}")

    print(f"Found {len(found_books)} books:")
    for book in found_books:
        print(f"- {book}")


import os

def validate_input_dir_structure(books_path):
    """
    Validate directory structure:

        books_root/
            <language>/
                <book_name>/
                    book_name.pdf

    Language folder names are arbitrary but the hierarchy depth is required.

    Returns:
        (bool, list[str]): validity flag and list of issues.
    """

    issues = []

    for language in os.listdir(books_path):
        language_path = os.path.join(books_path, language)

        if not os.path.isdir(language_path):
            continue

        # level 2 → books
        for item in os.listdir(language_path):
            book_path = os.path.join(language_path, item)

            # PDFs directly inside language folder → invalid
            if os.path.isfile(book_path) and item.lower().endswith(".pdf"):
                issues.append(
                    f"PDF outside book folder: {language}/{item}"
                )
                continue

            if not os.path.isdir(book_path):
                continue

            pdfs = [
                f for f in os.listdir(book_path)
                if f.lower().endswith(".pdf")
            ]

            if len(pdfs) == 0:
                issues.append(
                    f"No PDF found in book folder: {language}/{item}"
                )
                continue

            if len(pdfs) > 1:
                issues.append(
                    f"Multiple PDFs in book folder: {language}/{item}"
                )
                continue

            pdf_name = os.path.splitext(pdfs[0])[0]

            if pdf_name != item:
                issues.append(
                    f"PDF name does not match folder name: "
                    f"{language}/{item}/{pdfs[0]}"
                )

    return len(issues) == 0, issues


