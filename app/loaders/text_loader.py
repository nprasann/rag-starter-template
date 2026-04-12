from pathlib import Path

def load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def load_text_files_from_folder(folder_path: str):
    folder = Path(folder_path)
    documents = []

    for file_path in folder.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        documents.append({
            "filename": file_path.name,
            "content": content
        })

    return documents