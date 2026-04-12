from pypdf import PdfReader

def load_pdf_file(path: str) -> str:
    """
    Load text from a PDF file by extracting text from each page.

    Returns one combined text string for the whole PDF.
    """
    reader = PdfReader(path)
    parts = []

    for page in reader.pages:
        # extract_text() may return None on some pages
        text = page.extract_text() or ""
        parts.append(text)

    return "\n".join(parts)