"""
settings.py

Central configuration for Bhagavatham AI.

All configurable values used throughout the application
should be defined here.
"""

from pathlib import Path

# ==========================================================
# APPLICATION
# ==========================================================

APP_NAME = "Bhagavatham AI"

APP_VERSION = "1.0.0"

AUTHOR = "Vijay Viswanathan"
COPYRIGHT = "© 2026 Vijay Viswanathan"
# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

REPORTS_DIR = PROJECT_ROOT / "reports"

LOG_DIR = PROJECT_ROOT / "logs"

# ==========================================================
# CLEANER CONFIGURATION
# ==========================================================

MIN_PARAGRAPH_LENGTH = 15

MAX_HEADING_LENGTH = 100

MAX_HEADING_WORDS = 12

# Internet Archive navigation text to remove
ARCHIVE_REMOVE_LINES = [
    "Skip to main content",
    "Texts",
    "Video",
    "Audio",
    "Software",
    "Images",
    "Sign up for free",
    "Log in",
    "About",
    "Blog",
    "Events",
    "Projects",
    "Help",
    "Donate",
    "Contact",
    "Jobs",
    "Volunteer",
    "See other formats",
]

# Beginning of actual scripture
ARCHIVE_START_MARKERS = [
    "BOOK ONE",
    "BOOK NINE",
]

# Reserved for future use
ARCHIVE_END_MARKERS = []

# ==========================================================
# CHUNKING
# ==========================================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 150

MIN_CHUNK_SIZE = 300

# ==========================================================
# EMBEDDING MODEL
# ==========================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ==========================================================
# VECTOR DATABASE
# ==========================================================

CHROMA_COLLECTION = "bhagavatham"

# ==========================================================
# LLM
# ==========================================================

LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# ==========================================================
# RETRIEVAL
# ==========================================================

TOP_K = 5

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL = "INFO"
# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_SEED = 42
# ==========================================================
# ENCODING
# ==========================================================

DEFAULT_ENCODING = "utf-8"
CORPUS_REPORT_TXT = REPORTS_DIR / "corpus_report.txt"

CORPUS_REPORT_JSON = REPORTS_DIR / "corpus_report.json"
EXPECTED_SOURCE_DOCUMENTS = 2
# ======================================================
# Chunking Configuration
# ======================================================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150

MIN_CHUNK_SIZE = 200
CHUNKS_DIR = DATA_DIR / "chunks"