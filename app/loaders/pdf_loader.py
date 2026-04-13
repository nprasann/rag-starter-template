from pypdf import PdfReader
import logging

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
        logging.error(f"Failed to read PDF: {path}. Error: {e}")
        return ""