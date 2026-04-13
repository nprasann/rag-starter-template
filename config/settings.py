from pathlib import Path

# -----------------------------
# Folder configuration
# -----------------------------
# Folder containing source documents for ingestion
DATA_FOLDER = "data"

# Folder where generated result files are written
OUTPUT_FOLDER = "outputs"

# Folder where persistent ChromaDB data is stored locally
CHROMA_DB_PATH = "chroma_db"

# -----------------------------
# Default RAG settings
# -----------------------------
# Default Chroma collection name
DEFAULT_COLLECTION_NAME = "rag_demo"

# Default chunk size for splitting documents
DEFAULT_CHUNK_SIZE = 200

# Default overlap between chunks
DEFAULT_OVERLAP = 30

# Default number of chunks to retrieve
DEFAULT_TOP_K = 3

# Default output file path
DEFAULT_RESULT_FILE = Path(OUTPUT_FOLDER) / "result.md"