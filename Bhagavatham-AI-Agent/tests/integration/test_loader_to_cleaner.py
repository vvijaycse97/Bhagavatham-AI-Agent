"""
Integration Test
================

Pipeline Tested:

DocumentLoader
        │
        ▼
Document
        │
        ▼
TextCleaner
        │
        ▼
CleanDocument

This verifies that the loader and cleaner work together.
"""

import unittest

from rag.loader import DocumentLoader
from rag.cleaner import TextCleaner

from config.settings import RAW_DATA_DIR

from models.document import Document
from models.clean_document import CleanDocument


class TestLoaderToCleaner(unittest.TestCase):
    """
    Integration tests for Loader -> Cleaner pipeline.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = DocumentLoader(RAW_DATA_DIR)
        cls.cleaner = TextCleaner()

        cls.documents = cls.loader.load_documents()

        cls.results = [
            cls.cleaner.clean(document)
            for document in cls.documents
        ]

    def test_documents_loaded(self):
        """
        Loader should return documents.
        """
        self.assertGreater(len(self.documents), 0)

    def test_loaded_objects_are_documents(self):
        """
        Every loaded object must be a Document.
        """
        for document in self.documents:
            self.assertIsInstance(document, Document)

    def test_cleaner_returns_clean_document(self):
        """
        Cleaner should return CleanDocument objects.
        """
        for result in self.results:
            self.assertIsInstance(result, CleanDocument)

    def test_cleaned_text_exists(self):
        """
        Cleaned text should not be empty.
        """
        for result in self.results:
            self.assertTrue(result.cleaned_text.strip())

    def test_cleaned_text_smaller_or_equal(self):
        """
        Cleaning should never increase document size.
        """
        for result in self.results:
            self.assertLessEqual(
                len(result.cleaned_text),
                len(result.document.text),
            )

    def test_statistics_created(self):
        """
        CleanResult statistics should exist.
        """
        for result in self.results:
            self.assertIsNotNone(result.clean_result)

    def test_removed_characters_non_negative(self):
        """
        Removed character count should never be negative.
        """
        for result in self.results:
            self.assertGreaterEqual(
                result.clean_result.removed_characters,
                0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)