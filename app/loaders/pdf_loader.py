from pypdf import PdfReader


def load_pdf_file(path: str) -> str:
    """
    Load text from a PDF file by extracting text from each page.

    Returns one combined text string for the whole PDF.
    """
    try:
        reader = PdfReader(path)
        parts = []

        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)

        return "\n".join(parts)

    except Exception as e:
        print(f"Failed to read PDF: {path}. Error: {e}")
        return ""