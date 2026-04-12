from dotenv import load_dotenv
from pathlib import Path

from app.loaders.text_loader import load_text_file
from app.chunkers.simple_chunker import chunk_text
from app.embeddings.openai_embedder import embed_texts
from app.vectorstores.chroma_store import get_collection, index_chunks, search
from app.retrieval.qa import answer_question

def main():
    load_dotenv()

    text = load_text_file("data/sample.txt")
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    print("\nLoaded text and created chunks:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}:\n{chunk}")

    chunk_embeddings = embed_texts(chunks)

    collection = get_collection("rag_demo")
    index_chunks(collection, chunks, chunk_embeddings)

    question = input("\nEnter your question: ").strip()
    if not question:
        question = "What does RAG do?"

    query_embedding = embed_texts([question])[0]
    results = search(collection, query_embedding, top_k=3)

    retrieved_chunks = results["documents"][0]

    print("\nRetrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nRetrieved {i}:\n{chunk}")

    final_answer = answer_question(question, retrieved_chunks)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(final_answer)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/result.md").write_text(
        f"# RAG Result\n\n"
        f"## Question\n{question}\n\n"
        f"## Retrieved Chunks\n" +
        "\n\n".join([f"- {chunk}" for chunk in retrieved_chunks]) +
        f"\n\n## Answer\n{final_answer}\n",
        encoding="utf-8"
    )

    print("\nSaved output to outputs/result.md")

if __name__ == "__main__":
    main()