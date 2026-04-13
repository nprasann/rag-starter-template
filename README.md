# rag-starter-template

A minimal starter template for Retrieval-Augmented Generation (RAG) using Python, OpenAI embeddings, ChromaDB, and OpenAI responses.

## Features
- text loading
- PDF loading
- simple chunking
- embeddings
- vector indexing
- retrieval
- answer generation
- source tracking
- markdown output
- persistent local vector storage with ChromaDB
- skips re-indexing when persistent ChromaDB data already exists

## Supported Input Types
- `.txt`
- `.pdf`

## Project Structure
- `main.py` - runs the full RAG flow
- `app/loaders/text_loader.py` - loads supported documents from the data folder
- `app/loaders/pdf_loader.py` - extracts text from PDF files
- `app/chunkers/simple_chunker.py` - splits text into chunks
- `app/embeddings/openai_embedder.py` - creates embeddings
- `app/vectorstores/chroma_store.py` - stores and searches chunks
- `app/retrieval/qa.py` - generates grounded answers
- `data/` - folder containing source documents
- `outputs/result.md` - generated output after running

## Setup
1. Create and activate a virtual environment
2. Install dependencies:
   `pip install -r requirements.txt`
3. Create a `.env` file from `.env.example`
4. Add your OpenAI API key
5. Put `.txt` and/or `.pdf` files in the `data/` folder
6. Run:
   `python main.py`

## Notes
- PDF text extraction works best for text-based PDFs
- Scanned PDFs may require OCR before they work well in RAG
- ChromaDB data is stored locally in the `chroma_db/` folder
- The `chroma_db/` folder is ignored by Git because it is generated locally
- On first run, documents are indexed into ChromaDB
- On later runs, the app reuses the existing local vector store
- If you change files in `data/`, delete `chroma_db/` and run again to rebuild the index

## Next Steps
- add markdown loader
- support multiple file types
- add metadata filtering
- add configurable chunk sizes
- add persistent Chroma storage
- add evaluation harness
- add MCP server integration