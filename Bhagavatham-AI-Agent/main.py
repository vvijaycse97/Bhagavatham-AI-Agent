"""
Bhagavatham AI Agent

Main entry point for the application.

Current Stage:
    Stage 1 - Corpus Preparation

Future Stages:
    Stage 2 - Chunking
    Stage 3 - Embeddings
    Stage 4 - Vector Database
    Stage 5 - Retrieval
    Stage 6 - RAG Question Answering
"""

from datetime import datetime
from pathlib import Path

from rag.corpus_builder import CorpusBuilder
from utils.logger import get_logger
from config.settings import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

logger = get_logger(__name__)


def print_banner():
    """Print application banner."""

    print("=" * 70)
    print("            Bhagavatham AI Agent")
    print("          Stage 1 - Corpus Builder")
    print("=" * 70)


def print_summary(stats):
    """Print pipeline execution summary."""

    print("\n" + "=" * 70)
    print("Pipeline Execution Summary")
    print("=" * 70)

    print(f"Documents Processed      : {stats.documents_processed}")
    print(f"Original Characters      : {stats.original_characters:,}")
    print(f"Cleaned Characters       : {stats.cleaned_characters:,}")
    print(f"Removed Characters       : {stats.removed_characters:,}")
    print(f"Noise Paragraphs Removed : {stats.removed_noise_paragraphs}")
    print(f"Processing Time          : {stats.processing_time_seconds:.2f} sec")

    if stats.original_characters > 0:
        reduction = (
            stats.removed_characters / stats.original_characters
        ) * 100

        print(f"Reduction Percentage     : {reduction:.2f}%")

    print("\nOutput Directories")
    print("-" * 70)
    print(f"Processed Data : {PROCESSED_DATA_DIR}")
    print(f"Reports        : {REPORTS_DIR}")

    print("=" * 70)
    print("Stage 1 completed successfully.")
    print("=" * 70)
    
def main():
    """Application entry point."""

    start_time = datetime.now()

    print_banner()

    logger.info("Starting Bhagavatham AI Agent...")

    try:

        builder = CorpusBuilder()

        statistics = builder.build()

        end_time = datetime.now()

        duration = end_time - start_time

        print_summary(statistics)

        print(f"\nExecution Time : {duration}")

        logger.info("Application completed successfully.")

    except Exception as ex:

        logger.exception("Application failed.")

        print("\nApplication terminated due to an unexpected error.")
        print(ex)


if __name__ == "__main__":
    main()