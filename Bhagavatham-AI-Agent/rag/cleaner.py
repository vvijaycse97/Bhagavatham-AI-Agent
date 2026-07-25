"""
cleaner.py

Text cleaning service for Bhagavatham AI.

Responsibilities
----------------
- Cleans a Document
- Produces a CleanDocument
- Logs processing statistics

All text transformation logic resides in
rag.cleaning_rules.py.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from models import (
    CleanDocument,
    CleanResult,
    Document,
)

from rag.cleaning_rules import (
    normalize_text,
    remove_archive_preamble,
    remove_archive_navigation,
    remove_control_characters,
    remove_noise_paragraphs,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """
    Cleans Bhagavatham documents.

    This class orchestrates the cleaning pipeline but does
    not implement any cleaning logic itself.
    """

    PIPELINE: tuple[Callable[[str], str], ...] = (
        normalize_text,
        remove_archive_preamble,
        remove_archive_navigation,
        remove_control_characters,
    )

    def clean(self, document: Document) -> CleanDocument:
        """
        Clean a document.

        Parameters
        ----------
        document : Document
            Raw source document.

        Returns
        -------
        CleanDocument
            Cleaned document with cleaning statistics.
        """

        logger.info(
            "Cleaning %s (%s characters)",
            document.file_name,
            f"{len(document.text):,}",
        )

        #
        # Empty document
        #
        if not document.text.strip():

            logger.warning(
                "Document '%s' is empty.",
                document.file_name,
            )

            return CleanDocument(
                document=document,
                cleaned_text="",
                clean_result=CleanResult(
                    original_characters=0,
                    cleaned_characters=0,
                    removed_characters=0,
                    removed_noise_paragraphs=0,
                    processing_time_seconds=0.0,
                ),
            )

        start_time = time.perf_counter()

        original_text = document.text

        original_characters = len(original_text)

        cleaned_text = original_text

        #
        # Execute cleaning pipeline
        #
        for step in self.PIPELINE:

            logger.debug("Executing %s()", step.__name__)

            cleaned_text = step(cleaned_text)

        #
        # Remove OCR noise paragraphs
        #
        cleaned_text, removed_noise = remove_noise_paragraphs(
            cleaned_text
        )

        cleaned_characters = len(cleaned_text)

        removed_characters = (
            original_characters - cleaned_characters
        )

        removal_percentage = (
            removed_characters
            / original_characters
            * 100
        ) if original_characters else 0.0

        elapsed = time.perf_counter() - start_time

        statistics = CleanResult(
            original_characters=original_characters,
            cleaned_characters=cleaned_characters,
            removed_characters=removed_characters,
            removed_noise_paragraphs=removed_noise,
            processing_time_seconds=elapsed,
        )

        logger.info(
            "Finished %s",
            document.file_name,
        )

        logger.info(
            "Original Characters : %s",
            f"{original_characters:,}",
        )

        logger.info(
            "Cleaned Characters  : %s",
            f"{cleaned_characters:,}",
        )

        logger.info(
            "Characters Removed  : %s",
            f"{removed_characters:,}",
        )

        logger.info(
            "Removed Percentage  : %.2f%%",
            removal_percentage,
        )

        logger.info(
            "Noise Paragraphs Removed : %d",
            removed_noise,
        )

        logger.info(
            "Processing Time : %.2f sec",
            elapsed,
        )

        return CleanDocument(
            document=document,
            cleaned_text=cleaned_text,
            clean_result=statistics,
        )


if __name__ == "__main__":

    from rag.loader import DocumentLoader

    loader = DocumentLoader()

    cleaner = TextCleaner()

    documents = loader.load_documents()

    print("\n" + "=" * 80)
    print("Bhagavatham AI - Text Cleaner")
    print("=" * 80)

    for document in documents:

        clean_document = cleaner.clean(document)

        stats = clean_document.clean_result

        print(f"\nDocument : {document.file_name}")

        print("-" * 80)

        print(
            f"Original Characters : "
            f"{stats.original_characters:,}"
        )

        print(
            f"Cleaned Characters  : "
            f"{stats.cleaned_characters:,}"
        )

        print(
            f"Characters Removed  : "
            f"{stats.removed_characters:,}"
        )

        removal_percentage = (
            stats.removed_characters
            / stats.original_characters
            * 100
        ) if stats.original_characters else 0.0

        print(
            f"Removed Percentage  : "
            f"{removal_percentage:.2f}%"
        )

        print(
            f"Noise Paragraphs    : "
            f"{stats.removed_noise_paragraphs:,}"
        )

        print(
            f"Processing Time     : "
            f"{stats.processing_time_seconds:.2f} sec"
        )

        print("\nPreview")
        print("-" * 80)

        print(clean_document.cleaned_text[:600])

        print("\n" + "=" * 80)