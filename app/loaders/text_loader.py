from pathlib import Path
from app.loaders.pdf_loader import load_pdf_file

def load_text_file(path: str) -> str:
    """
    Load a plain text file and return its contents.
    """
    return Path(path).read_text(encoding="utf-8")

def load_documents_from_folder(folder_path: str):
    """
    Load supported files from a folder.

    Currently supports:
    - .txt
    - .pdf

    Returns:
        A list of dicts like:
        {
            "filename": "...",
            "content": "..."
        }
    """
    folder = Path(folder_path)
    documents = []

    # Load text files
    for file_path in folder.glob("*.txt"):
        content = load_text_file(str(file_path))
        documents.append({
            "filename": file_path.name,
            "content": content
        })

    # Load PDF files
    for file_path in folder.glob("*.pdf"):
        content = load_pdf_file(str(file_path))
        documents.append({
            "filename": file_path.name,
            "content": content
        })

    return documents