import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from config.settings import (
    DATA_FOLDER,
    OUTPUT_FOLDER,
    CHROMA_DB_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
)
from app.loaders.text_loader import load_documents_from_folder
from app.chunkers.simple_chunker import chunk_text
from app.embeddings.openai_embedder import embed_texts
from app.vectorstores.chroma_store import get_collection, index_chunks, search
from app.retrieval.qa import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

def limit_chunks_per_source(retrieved_chunks, retrieved_metadatas, max_per_source=1):
    """
    Limit how many retrieved chunks can come from the same source file.
    """
    source_counts = {}
    filtered_chunks = []
    filtered_metadatas = []

    for chunk, metadata in zip(retrieved_chunks, retrieved_metadatas):
        source = metadata["source"]
        current_count = source_counts.get(source, 0)

        if current_count < max_per_source:
            filtered_chunks.append(chunk)
            filtered_metadatas.append(metadata)
            source_counts[source] = current_count + 1

    return filtered_chunks, filtered_metadatas

def ensure_index_exists():
    """
    Build the index only if the collection is empty.
    """
    documents = load_documents_from_folder(DATA_FOLDER)
    if not documents:
        raise ValueError("No supported documents found in the data/ folder.")

    all_chunks = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_text(
            doc["content"],
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_OVERLAP
        )

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["filename"]})

    if not all_chunks:
        raise ValueError("Documents were loaded, but no chunks were created.")

    collection = get_collection(
        DEFAULT_COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH
    )

    existing_count = collection.count()

    if existing_count == 0:
        logging.info("No existing vectors found. Indexing documents now...")
        chunk_embeddings = embed_texts(all_chunks)
        index_chunks(collection, all_chunks, chunk_embeddings, all_metadatas)
        logging.info(f"Indexed {len(all_chunks)} chunk(s) into ChromaDB.")
    else:
        logging.info(f"Using existing ChromaDB collection with {existing_count} chunk(s).")

    return collection


def run_evaluation():
    """
    Run evaluation questions against the current RAG pipeline.
    """
    load_dotenv()
    collection = ensure_index_exists()

    questions_path = Path("evaluation/questions.json")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    lines = []
    lines.append("# Evaluation Results\n")

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])

        query_embedding = embed_texts([question])[0]
        results = search(collection, query_embedding, top_k=DEFAULT_TOP_K)

        retrieved_docs = results.get("documents", [])
        retrieved_meta = results.get("metadatas", [])

        retrieved_chunks = retrieved_docs[0] if retrieved_docs else []
        retrieved_metadatas = retrieved_meta[0] if retrieved_meta else []

        retrieved_chunks, retrieved_metadatas = limit_chunks_per_source(
            retrieved_chunks,
            retrieved_metadatas,
            max_per_source=1
        )

        retrieved_sources = []
        for metadata in retrieved_metadatas:
            source = metadata["source"]
            if source not in retrieved_sources:
                retrieved_sources.append(source)

        if retrieved_chunks:
            answer = answer_question(question, retrieved_chunks)
        else:
            answer = "I could not find relevant information in the indexed documents."

        lines.append(f"## Test {i}")
        lines.append(f"**Question:** {question}")
        lines.append(f"**Expected Sources:** {', '.join(expected_sources) if expected_sources else 'None'}")
        lines.append(f"**Retrieved Sources:** {', '.join(retrieved_sources) if retrieved_sources else 'None'}")
        lines.append(f"**Answer:** {answer}\n")

    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    output_path = Path(OUTPUT_FOLDER) / "evaluation_results.md"
    output_path.write_text("\n\n".join(lines), encoding="utf-8")

    print(f"\nSaved evaluation results to {output_path}")


if __name__ == "__main__":
    run_evaluation()