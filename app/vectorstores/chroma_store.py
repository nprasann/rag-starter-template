import chromadb

def get_collection(name="rag_demo"):
    client = chromadb.Client()
    return client.get_or_create_collection(name=name)

def index_chunks(collection, chunks, embeddings, metadatas):
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

def search(collection, query_embedding, top_k=3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )