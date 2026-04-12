# rag-starter-template

A minimal starter template for Retrieval-Augmented Generation (RAG) using Python, OpenAI embeddings, ChromaDB, and OpenAI responses.

## Features
- text loading
- simple chunking
- embeddings
- vector indexing
- retrieval
- answer generation
- markdown output
- Retrieval includes source attribution for transparency

## Project Structure
- `main.py` - runs the full RAG flow
- `app/loaders/text_loader.py` - loads text files
- `app/chunkers/simple_chunker.py` - splits text into chunks
- `app/embeddings/openai_embedder.py` - creates embeddings
- `app/vectorstores/chroma_store.py` - stores and searches chunks
- `app/retrieval/qa.py` - generates grounded answers
- `data/` - folder containing multiple text documents
- `outputs/result.md` - generated output after running

## Setup
1. Create and activate a virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Create a `.env` file from `.env.example`
4. Add your OpenAI API key
5. Run:
   python main.py

## Next Steps
- add markdown loader
- add pdf loader
- support multiple documents
- add metadata filtering
- add configurable chunk sizes
- add evaluation harness
- add MCP server integration
## 🆕 Enhancements

- Supports multiple documents from the `data/` folder
- Adds metadata (source file name) to each chunk
- Displays source of retrieved chunks during query