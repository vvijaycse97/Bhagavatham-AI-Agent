"""
test_loader.py

Unit tests for rag.loader.

Responsibilities
----------------
- Verify DocumentLoader initialization
- Verify raw data directory exists
- Verify documents are loaded
- Verify Document model integrity
"""

from pathlib import Path
import unittest

from config import (
    RAW_DATA_DIR,
    EXPECTED_SOURCE_DOCUMENTS,
)
from models import Document
from rag.loader import DocumentLoader


class TestDocumentLoader(unittest.TestCase):
    """
    Unit tests for DocumentLoader.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create one loader for all tests.
        """

        cls.loader = DocumentLoader()

        cls.documents = cls.loader.load_documents()

    # ---------------------------------------------------------
    # Loader Tests
    # ---------------------------------------------------------

    def test_loader_instance(self):
        """
        Loader should be instantiated successfully.
        """

        self.assertIsInstance(
            self.loader,
            DocumentLoader,
        )

    def test_raw_directory_exists(self):
        """
        Raw data directory should exist.
        """

        self.assertTrue(
            RAW_DATA_DIR.exists()
        )

        self.assertTrue(
            RAW_DATA_DIR.is_dir()
        )

    def test_documents_loaded(self):
        """
        Loader should return at least one document.
        """

        self.assertGreater(
            len(self.documents),
            0,
        )

    def test_expected_document_count(self):
        """
        Current corpus should contain exactly two documents.
        """

        self.assertEqual(
            len(self.documents),
            EXPECTED_SOURCE_DOCUMENTS,
        )

    # ---------------------------------------------------------
    # Document Model Tests
    # ---------------------------------------------------------

    def test_document_instance(self):
        """
        Every loaded object should be a Document.
        """

        for document in self.documents:

            self.assertIsInstance(
                document,
                Document,
            )

    def test_document_filename(self):
        """
        Every document should have a filename.
        """

        for document in self.documents:

            self.assertTrue(
                document.file_name.endswith(".txt")
            )

    def test_document_filepath(self):
        """
        Every document path should exist.
        """

        for document in self.documents:

            self.assertIsInstance(
                document.file_path,
                Path,
            )

            self.assertTrue(
                document.file_path.exists()
            )

    def test_document_text(self):
        """
        Every document should contain text.
        """

        for document in self.documents:

            self.assertGreater(
                len(document.text),
                0,
            )

    def test_document_not_blank(self):
        """
        Document text should not be whitespace.
        """

        for document in self.documents:

            self.assertNotEqual(
                document.text.strip(),
                "",
            )


if __name__ == "__main__":

    unittest.main(
        verbosity=2,
    )