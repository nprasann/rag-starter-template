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
- Markdown loading
- basic logging for indexing, loading, and error reporting
- optional retrieval filtering by source filename
- supports custom ChromaDB collection names
- graceful handling when no relevant chunks are found
- centralized configuration file for default RAG settings

## Supported Input Types
- `.txt`
- `.md`
- `.pdf`
## Command-Line Options
You can customize the RAG run with command-line flags:
```bash
python main.py --rebuild --chunk-size 250 --overlap 40 --top-k 4 What is RAG?
```
	
     --rebuild → rebuild the local ChromaDB index
	  --chunk-size → control chunk size
	  --overlap → control chunk overlap
     --top-k → number of chunks to retrieve
     --source → restrict retrieval to a specific file in the data/ folder
    `--collection-name` → choose which ChromaDB collection to use
   ```bash
      python main.py --collection-name policy_docs --rebuild What is RAG?
   ```
	

## Error Handling
This starter template includes basic error handling for:
- missing API key
- empty `data/` folder
- unreadable or empty documents
- PDF extraction failures
- Uses Python logging for workflow visibility and error messages
- If retrieval finds no relevant chunks, the app returns a friendly response and still writes `outputs/result.md`
- If `--source` is used and no results are found, the app logs that the source filter may be too restrictive

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
- `config/settings.py` - central location for default folders and RAG settings

## Setup
1. Create and activate a virtual environment
2. Install dependencies:
   `pip install -r requirements.txt`
3. Create a `.env` file from `.env.example`
4. Add your OpenAI API key
5. Put `.txt`, `.md`, and/or `.pdf` files in the `data/` folder
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
- Default settings such as chunk size, overlap, top-k, and folder paths are stored in `config/settings.py`
- Command-line arguments override config defaults at runtime
### Rebuilding the Index
If you change files in the `data/` folder and want to rebuild the vector index, run:
```bash
python main.py --rebuild
```
You can also rebuild and ask a question in one command:
```bash
python main.py --rebuild What is chunking?
```

## Next Steps
- add markdown loader
- support multiple file types
- add metadata filtering
- add configurable chunk sizes
- add persistent Chroma storage
- add evaluation harness
- add MCP server integration