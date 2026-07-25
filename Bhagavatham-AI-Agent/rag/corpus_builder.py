"""
corpus_builder.py

Builds the processed Bhagavatham corpus.

Responsibilities
----------------
- Load raw documents
- Clean documents
- Save cleaned corpus
- Aggregate corpus statistics
- Generate corpus reports

This module orchestrates the ingestion pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

from config import (
    DEFAULT_ENCODING,
    PROCESSED_DATA_DIR,
)

from models import (
    CleanDocument,
    CorpusStatistics,
)

from rag.cleaner import TextCleaner
from rag.loader import DocumentLoader
from rag.report_writer import ReportWriter

from utils import (
    get_clean_filename,
    get_logger,
)

logger = get_logger(__name__)


class CorpusBuilder:
    """
    Builds the processed Bhagavatham corpus.

    Pipeline

    Raw TXT
        ↓
    Loader
        ↓
    Cleaner
        ↓
    Save Clean Files
        ↓
    Report Writer
    """

    def __init__(self) -> None:

        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.loader = DocumentLoader()

        self.cleaner = TextCleaner()

        self.report_writer = ReportWriter()

        self.statistics = CorpusStatistics()
    def build(self) -> CorpusStatistics:
        """
        Execute the corpus build pipeline.

        Returns
        -------
        CorpusStatistics
        """

        logger.info("=" * 70)
        logger.info("Starting Corpus Builder")
        logger.info("=" * 70)

        start = time.perf_counter()

        documents = self.loader.load_documents()

        logger.info(
        "Processing %d document(s)...",
        len(documents),
        )

        for document in documents:

            clean_document = self.cleaner.clean(document)

            self._process_document(
            clean_document
           )

        self.statistics.processing_time_seconds = (
            time.perf_counter() - start
        )

        self.report_writer.write(
            self.statistics
        )

        logger.info(
            "Corpus build completed successfully."
        )

        return self.statistics

    def _save_clean_document(
        self,
        clean_document: CleanDocument,
    ) -> None:
        """
        Save a cleaned document to the processed directory.

        Parameters
        ----------
        clean_document : CleanDocument
        """

        output_file = (
            PROCESSED_DATA_DIR
            / get_clean_filename(
                clean_document.document.file_name
            )
        )

        logger.info(
            "Saving cleaned document: %s",
            output_file.name,
        )

        output_file.write_text(
            clean_document.cleaned_text,
            encoding=DEFAULT_ENCODING,
        )
        logger.debug(
        "Output file size: %s bytes",
        output_file.stat().st_size,
        )

        logger.debug(
            "Saved %s (%s characters)",
            output_file.name,
            f"{len(clean_document.cleaned_text):,}",
        )
    def _process_document(
        self,
        clean_document: CleanDocument,
    ) -> None:
        """
        Process one cleaned document.

        Responsibilities
        ----------------
        - Save cleaned text
        - Update corpus statistics
        """

        self._save_clean_document(
            clean_document
        )

        self._update_statistics(
            clean_document
        )
    def _update_statistics(
        self,
        clean_document: CleanDocument,
    ) -> None:
        """
        Update aggregate corpus statistics.

        Parameters
        ----------
        clean_document : CleanDocument
        """

        result = clean_document.clean_result

        self.statistics.documents_processed += 1

        self.statistics.original_characters += (
            result.original_characters
        )

        self.statistics.cleaned_characters += (
            result.cleaned_characters
        )

        self.statistics.removed_characters += (
            result.removed_characters
        )

        self.statistics.removed_noise_paragraphs += (
            result.removed_noise_paragraphs
        )

if __name__ == "__main__":

    builder = CorpusBuilder()

    statistics = builder.build()

    print("\n" + "=" * 80)
    print("Bhagavatham AI - Corpus Builder")
    print("=" * 80)

    print(
        f"Documents Processed      : "
        f"{statistics.documents_processed:,}"
    )

    print(
        f"Original Characters      : "
        f"{statistics.original_characters:,}"
    )

    print(
        f"Cleaned Characters       : "
        f"{statistics.cleaned_characters:,}"
    )

    print(
        f"Characters Removed       : "
        f"{statistics.removed_characters:,}"
    )

    print(
        f"Noise Paragraphs Removed : "
        f"{statistics.removed_noise_paragraphs:,}"
    )

    print(
        f"Processing Time          : "
        f"{statistics.processing_time_seconds:.2f} sec"
    )

    print("\nProcessed files saved to:")

    print(PROCESSED_DATA_DIR)

    print("\nReports generated in:")

    print(Path("reports").resolve())

    print("\n" + "=" * 80)