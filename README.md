# rag-starter-template

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-v1--complete-success)
![RAG](https://img.shields.io/badge/AI-RAG-orange)

A minimal, production-ready starter template for **Retrieval-Augmented Generation (RAG)** using Python, OpenAI embeddings, ChromaDB, and OpenAI responses.

---

## 🚀 Features

### 🧠 Core RAG Pipeline
- Multi-format document loading (`.txt`, `.md`, `.pdf`)
- Text chunking with configurable size and overlap
- Embedding generation using OpenAI
- Vector storage and retrieval with ChromaDB
- Answer generation using retrieved context
- Source tracking and citation support

### ⚡ Performance & Storage
- Persistent local vector storage (ChromaDB)
- Skips re-indexing if data already exists
- Rebuild support via CLI flag

### 🔍 Retrieval Enhancements
- Metadata filtering by source file
- Duplicate control (limit chunks per source)
- Custom collection names for multiple indexes

### 🛠️ Usability
- Command-line configuration
- Centralized configuration file
- Markdown output (`outputs/result.md`)
- Graceful handling of no-result scenarios

### 📊 Observability & Quality
- Structured logging
- Evaluation harness
- PASS / FAIL / CHECK reporting

---

## 📂 Supported Input Types

- `.txt`
- `.md`
- `.pdf`

---

## ⚙️ Command-Line Options

Run:

```bash
python main.py --rebuild --chunk-size 250 --overlap 40 --top-k 4 What is RAG?
```

### Available flags

| Option | Description |
|--------|-------------|
| `--rebuild` | Rebuild the local ChromaDB index |
| `--chunk-size` | Control chunk size |
| `--overlap` | Control chunk overlap |
| `--top-k` | Number of chunks to retrieve |
| `--source` | Restrict retrieval to a specific file |
| `--collection-name` | Use a custom ChromaDB collection |
| `--max-per-source` | Limit chunks per source file |

### Examples

```bash
python main.py --collection-name policy_docs --rebuild What is RAG?
python main.py --max-per-source 2 What are the common steps in a RAG pipeline?
```

---

## 🛡️ Error Handling

Handles common failure scenarios gracefully:

- Missing API key
- Empty `data/` folder
- Unreadable or empty documents
- PDF extraction failures
- No relevant retrieval results

Additional behavior:
- Logs warnings and errors using Python logging
- Returns a friendly fallback answer if no results are found
- Still generates `outputs/result.md` even on failure cases

---

## 🏗️ Project Structure

```text
main.py                        # Main RAG pipeline
app/
  loaders/
    text_loader.py            # Load txt/md files
    pdf_loader.py             # Load PDF files
  chunkers/
    simple_chunker.py         # Text chunking
  embeddings/
    openai_embedder.py        # Embedding generation
  vectorstores/
    chroma_store.py           # Vector DB operations
  retrieval/
    qa.py                     # Answer generation

config/
  settings.py                 # Central configuration

evaluation/
  questions.json              # Test questions
  run_eval.py                 # Evaluation runner

data/                         # Input documents
outputs/                      # Generated results
```

---

## ⚡ Setup

1. Create and activate a virtual environment  
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`  
4. Add your OpenAI API key  
5. Add documents to `data/`  
6. Run:

```bash
python main.py
```

---

## 🔄 Rebuilding the Index

If you change documents:

```bash
python main.py --rebuild
```

Or:

```bash
python main.py --rebuild What is chunking?
```

---

## 🧠 Evaluation Harness

Run:

```bash
python -m evaluation.run_eval
```

Output:

```
outputs/evaluation_results.md
```

Includes:
- Question
- Expected sources
- Retrieved sources
- Matched sources
- Generated answer
- PASS / FAIL / CHECK status
- Summary counts

---

## 📌 Notes

- PDF extraction works best for text-based PDFs  
- Scanned PDFs may require OCR  
- ChromaDB data is stored in `chroma_db/` (ignored by Git)  
- First run builds the index, later runs reuse it  
- CLI arguments override config defaults  
- Config values are stored in `config/settings.py`  

---

## 🧭 Roadmap

### Near-term
- Improve evaluation scoring
- Enhance configuration flexibility

### Future
- Support additional file types (`.docx`, `.csv`, `.json`)
- Add hybrid search (keyword + vector)
- Add MCP server integration
- Support multiple vector stores (FAISS, Pinecone, etc.)

---

## ⭐ If you found this useful

Give the repo a star ⭐ and feel free to fork or extend it.
