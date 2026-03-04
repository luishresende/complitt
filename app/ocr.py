from tqdm import tqdm

from app.utils import validate_input_dir_structure

import os

def run_paddle(books_path, relative_output_dir="paddle_output"):
    """
    Find all PDFs under `books_path`, then run PaddleOCR on each PDF.
    For each PDF, create a 'paddle_output' folder in the same directory
    and save OCR results there.

    Args:
        books_path (str): Root directory containing PDFs.
        relative_output_dir (str, optional): Relative path to save OCR results. Defaults to "paddle_output".
    """
    from paddleocr import PaddleOCRVL
    import torch
    # Validate directory structure
    valid_dir, issues = validate_input_dir_structure(books_path)
    if not valid_dir:
        print("Invalid directory structure.")
        for issue in issues:
            print(issue)
        print(f"Please fix it running the command:\nrun.py prepare_dir {books_path}")
        return

    # Automatic device selection
    device = "gpu:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize OCR pipeline
    pipeline = PaddleOCRVL(device=device)

    # Collect all PDF paths
    pdf_list = []
    for root, dirs, files in os.walk(books_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_list.append(os.path.join(root, file))

    if not pdf_list:
        print("No PDFs found.")
        return

    # Process PDFs
    for pdf_path in tqdm(pdf_list, desc="Processing PDFs", unit="file"):
        output_dir = os.path.join(os.path.dirname(pdf_path), "paddle_output")
        os.makedirs(output_dir, exist_ok=True)

        output = pipeline.predict(pdf_path)

        for res in output:
            res.print()
            res.save_to_json(save_path=output_dir)
            res.save_to_markdown(save_path=output_dir)
