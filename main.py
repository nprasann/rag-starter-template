from dotenv import load_dotenv
from pathlib import Path
import shutil
import argparse
import os
import logging 

from config.settings import (
    DATA_FOLDER,
    OUTPUT_FOLDER,
    CHROMA_DB_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_RESULT_FILE,
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

    Args:
        retrieved_chunks: list of retrieved text chunks
        retrieved_metadatas: list of metadata dicts for the chunks
        max_per_source: maximum number of chunks allowed per source file

    Returns:
        filtered_chunks, filtered_metadatas
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

def main():
    # Load environment variables (.env) so API key is available
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("Missing OPENAI_API_KEY.")
        logging.info("Create a .env file with your OpenAI API key and try again.")
        return

    # -----------------------------
    # CONFIG: Command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Run the RAG starter template")
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=1,
        help="Maximum number of retrieved chunks allowed from the same source file"
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the local ChromaDB index")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for splitting documents")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Chunk overlap size")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("question", nargs="*", help="Optional question to ask")
    parser.add_argument("--source", type=str, help="Optional source file to filter retrieval by")
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME, help="Name of the ChromaDB collection to use")
    args = parser.parse_args()

    try:
        # -----------------------------
        # OPTIONAL: Rebuild vector DB
        # -----------------------------
        if args.rebuild and Path(CHROMA_DB_PATH).exists():
            logging.info("Rebuild flag detected. Deleting existing ChromaDB...")
            shutil.rmtree(CHROMA_DB_PATH)

        # -----------------------------
        # STEP 1: Load supported documents
        # -----------------------------
        documents = load_documents_from_folder(DATA_FOLDER)

        if not documents:
            logging.warning("No supported documents found in the data/ folder.")
            logging.info("Add at least one .txt, .md, or .pdf file and try again.")
            return

        all_chunks = []
        all_metadatas = []

        # -----------------------------
        # STEP 2: Chunk documents + attach metadata
        # -----------------------------
        for doc in documents:
            chunks = chunk_text(
                doc["content"],
                chunk_size=args.chunk_size,
                overlap=args.overlap
            )

            for chunk in chunks:
                all_chunks.append(chunk)

                # Track which file each chunk came from
                all_metadatas.append({"source": doc["filename"]})

        if not all_chunks:
            logging.warning("Documents were loaded, but no chunks were created.")
            logging.info("Check whether the files contain readable text.")
            return

        logging.info(f"Loaded {len(documents)} document(s)")
        logging.info(f"Created {len(all_chunks)} chunk(s)")

        # -----------------------------
        # STEP 3: Open vector DB
        # -----------------------------
        logging.info(f"Using collection: {args.collection_name}")
        collection = get_collection(args.collection_name, persist_directory=CHROMA_DB_PATH)

        # Check whether the collection already has indexed data
        existing_count = collection.count()

        if existing_count == 0:
            logging.info("No existing vectors found. Indexing documents now...")

            # Convert text chunks into vectors using OpenAI embeddings
            chunk_embeddings = embed_texts(all_chunks)

            # Store chunks + embeddings + metadata in ChromaDB
            index_chunks(collection, all_chunks, chunk_embeddings, all_metadatas)
            logging.info(f"Indexed {len(all_chunks)} chunk(s) into ChromaDB.")
        else:
            logging.info(f"Using existing ChromaDB collection with {existing_count} chunk(s).")

        # -----------------------------
        # STEP 4: Get user question
        # -----------------------------
        if args.question:
            question = " ".join(args.question).strip()
        else:
            question = input("\nEnter your question: ").strip()

        if not question:
            question = "What does RAG do?"

        # -----------------------------
        # STEP 5: Retrieve relevant chunks
        # -----------------------------
        query_embedding = embed_texts([question])[0]

        results = search(
            collection,
            query_embedding,
            top_k=args.top_k,
            source_filter=args.source
        )

        # Safely extract results
        retrieved_docs = results.get("documents", [])
        retrieved_meta = results.get("metadatas", [])

        retrieved_chunks = retrieved_docs[0] if retrieved_docs else []
        retrieved_metadatas = retrieved_meta[0] if retrieved_meta else []

        # Limit repeated chunks from the same source file
        retrieved_chunks, retrieved_metadatas = limit_chunks_per_source(
            retrieved_chunks,
            retrieved_metadatas,
            max_per_source=args.max_per_source
        )
        logging.info(
            f"Applied duplicate control with max_per_source={args.max_per_source}"
        )

        # Handle no results
        if not retrieved_chunks:
            logging.warning("No matching chunks found for the question.")

            if args.source:
                logging.info(f"No results were found with source filter: {args.source}")

            print("\nQuestion:")
            print(question)

            print("\nAnswer:")
            print("I could not find relevant information in the indexed documents.")

            Path("outputs").mkdir(exist_ok=True)
            DEFAULT_RESULT_FILE.write_text(
                "# RAG Result\n\n"
                f"## Question\n{question}\n\n"
                "## Answer\n"
                "I could not find relevant information in the indexed documents.\n",
                encoding="utf-8"
            )

            print("\nSaved output to outputs/result.md")
            return
        
        results = search(
            collection,
            query_embedding,
            top_k=args.top_k,
            source_filter=args.source
        )

        retrieved_docs = results.get("documents", [])
        retrieved_meta = results.get("metadatas", [])

        retrieved_chunks = retrieved_docs[0] if retrieved_docs else []
        retrieved_metadatas = retrieved_meta[0] if retrieved_meta else []

        # Handle cases where retrieval finds no useful chunks
        if not retrieved_chunks:
            logging.warning("No matching chunks found for the question.")

            print("\nQuestion:")
            print(question)

            print("\nAnswer:")
            print("I could not find relevant information in the indexed documents.")

            Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
            DEFAULT_RESULT_FILE.write_text(
                "# RAG Result\n\n"
                f"## Question\n{question}\n\n"
                "## Answer\n"
                "I could not find relevant information in the indexed documents.\n",
                encoding="utf-8"
            )

            print("\nSaved output to outputs/result.md")
            return

        print("\nRetrieved chunks:")
        for i, (chunk, metadata) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1):
            print(f"\nRetrieved {i} (source: {metadata['source']}):\n{chunk}")

        # -----------------------------
        # STEP 6: Generate final answer
        # -----------------------------
        final_answer = answer_question(question, retrieved_chunks)

        print("\nQuestion:")
        print(question)

        print("\nAnswer:")
        print(final_answer)

        # -----------------------------
        # STEP 7: Save results to file
        # -----------------------------
        Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

        lines = []
        lines.append("# RAG Result\n")
        lines.append(f"## Question\n{question}\n")
        lines.append("## Retrieved Chunks\n")

        for i, (chunk, metadata) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1):
            lines.append(f"### Retrieved {i}")
            lines.append(f"**Source:** {metadata['source']}")
            lines.append(f"**Chunk:** {chunk}\n")

        lines.append("## Answer\n")
        lines.append(final_answer)

        seen_sources = []
        for metadata in retrieved_metadatas:
            source = metadata["source"]
            if source not in seen_sources:
                seen_sources.append(source)

        lines.append("\n## Sources\n")
        for source in seen_sources:
            lines.append(f"- {source}")

        Path("outputs/result.md").write_text("\n".join(lines), encoding="utf-8")
        print("\nSaved output to outputs/result.md")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()