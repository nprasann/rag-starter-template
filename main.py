from dotenv import load_dotenv
from pathlib import Path
import sys

from app.loaders.text_loader import load_documents_from_folder
from app.chunkers.simple_chunker import chunk_text
from app.embeddings.openai_embedder import embed_texts
from app.vectorstores.chroma_store import get_collection, index_chunks, search
from app.retrieval.qa import answer_question


def main():
    # Load environment variables (.env) so API key is available
    load_dotenv()

    # -----------------------------
    # STEP 1: Load supported documents (.txt and .pdf)
    # -----------------------------

    # Load all supported documents from the data folder
    documents = load_documents_from_folder("data")

    all_chunks = []
    all_metadatas = []

    # -----------------------------
    # STEP 2: Chunk documents + attach metadata
    # -----------------------------
    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size=200, overlap=30)

        for chunk in chunks:
            all_chunks.append(chunk)

            # Track which file each chunk came from
            all_metadatas.append({"source": doc["filename"]})

    print(f"\nLoaded {len(documents)} document(s)")
    print(f"Created {len(all_chunks)} chunk(s)")

        # -----------------------------
    # STEP 3: Open vector DB
    # -----------------------------
    collection = get_collection("rag_demo")

    # Check whether the collection already has indexed data
    existing_count = collection.count()

    if existing_count == 0:
        print("\nNo existing vectors found. Indexing documents now...")

        # Convert text chunks into vectors using OpenAI embeddings
        chunk_embeddings = embed_texts(all_chunks)

        # Store chunks + embeddings + metadata in ChromaDB
        index_chunks(collection, all_chunks, chunk_embeddings, all_metadatas)

        print(f"Indexed {len(all_chunks)} chunk(s) into ChromaDB.")
    else:
        print(f"\nUsing existing ChromaDB collection with {existing_count} chunk(s).")
    # -----------------------------
    # STEP 5: Get user question
    # -----------------------------
    # Support CLI input OR interactive input
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("\nEnter your question: ").strip()

    # Default fallback question
    if not question:
        question = "What does RAG do?"

    # -----------------------------
    # STEP 6: Retrieve relevant chunks
    # -----------------------------
    # Convert question to embedding
    query_embedding = embed_texts([question])[0]

    # Search vector DB for similar chunks
    results = search(collection, query_embedding, top_k=3)

    retrieved_chunks = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    print("\nRetrieved chunks:")
    for i, (chunk, metadata) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1):
        print(f"\nRetrieved {i} (source: {metadata['source']}):\n{chunk}")

    # -----------------------------
    # STEP 7: Generate final answer
    # -----------------------------
    # Send retrieved chunks + question to LLM
    final_answer = answer_question(question, retrieved_chunks)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(final_answer)

    # -----------------------------
    # STEP 8: Save results to file
    # -----------------------------
    Path("outputs").mkdir(exist_ok=True)

    lines = []
    lines.append("# RAG Result\n")
    lines.append(f"## Question\n{question}\n")
    lines.append("## Retrieved Chunks\n")

    # Save retrieved chunks with source info
    for i, (chunk, metadata) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1):
        lines.append(f"### Retrieved {i}")
        lines.append(f"**Source:** {metadata['source']}")
        lines.append(f"**Chunk:** {chunk}\n")

    # Save final answer
    lines.append("## Answer\n")
    lines.append(final_answer)

    # Save unique sources used
    seen_sources = []
    for metadata in retrieved_metadatas:
        source = metadata["source"]
        if source not in seen_sources:
            seen_sources.append(source)

    lines.append("\n## Sources\n")
    for source in seen_sources:
        lines.append(f"- {source}")

    # Write output file
    Path("outputs/result.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nSaved output to outputs/result.md")


if __name__ == "__main__":
    main()