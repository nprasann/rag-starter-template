from dotenv import load_dotenv
from pathlib import Path

from app.loaders.text_loader import load_text_files_from_folder
from app.chunkers.simple_chunker import chunk_text
from app.embeddings.openai_embedder import embed_texts
from app.vectorstores.chroma_store import get_collection, index_chunks, search
from app.retrieval.qa import answer_question

def main():
    load_dotenv()

    documents = load_text_files_from_folder("data")

    all_chunks = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size=200, overlap=30)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["filename"]})

    print(f"\nLoaded {len(documents)} document(s)")
    print(f"Created {len(all_chunks)} chunk(s)")

    chunk_embeddings = embed_texts(all_chunks)

    collection = get_collection("rag_demo")
    index_chunks(collection, all_chunks, chunk_embeddings, all_metadatas)

    question = input("\nEnter your question: ").strip()
    if not question:
        question = "What does RAG do?"

    query_embedding = embed_texts([question])[0]
    results = search(collection, query_embedding, top_k=3)

    retrieved_chunks = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    print("\nRetrieved chunks:")
    for i, (chunk, metadata) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1):
        print(f"\nRetrieved {i} (source: {metadata['source']}):\n{chunk}")

    final_answer = answer_question(question, retrieved_chunks)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(final_answer)

    Path("outputs").mkdir(exist_ok=True)

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

    Path("outputs/result.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nSaved output to outputs/result.md")

if __name__ == "__main__":
    main()