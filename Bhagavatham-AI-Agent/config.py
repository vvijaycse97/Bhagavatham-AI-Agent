from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 5