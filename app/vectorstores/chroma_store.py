import chromadb
from chromadb.config import Settings


def get_collection(name="rag_demo", persist_directory="chroma_db"):
    """
    Create or load a persistent ChromaDB collection.

    - If the database already exists on disk, it will be reused.
    - If not, a new one will be created.

    persist_directory:
        Folder where vector data is stored locally.
    """

    # Create a persistent ChromaDB client (stored on disk)
    client = chromadb.PersistentClient(path=persist_directory)

    # Get existing collection or create a new one
    return client.get_or_create_collection(name=name)


def index_chunks(collection, chunks, embeddings, metadatas):
    """
    Store chunks, embeddings, and metadata in the vector database.

    Each chunk is assigned a unique ID and stored along with:
    - original text (document chunk)
    - embedding vector
    - metadata (e.g., source filename)
    """

    # Generate unique IDs for each chunk
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    # Add all data into ChromaDB
    collection.add(
        ids=ids,
        documents=chunks,      # original text chunks
        embeddings=embeddings, # vector representations
        metadatas=metadatas   # additional info like source file
    )


def search(collection, query_embedding, top_k=3):
    """
    Perform similarity search using the query embedding.

    Returns:
        The top_k most relevant chunks based on semantic similarity.
    """

    return collection.query(
        query_embeddings=[query_embedding],  # vector of user question
        n_results=top_k                      # number of results to return
    )