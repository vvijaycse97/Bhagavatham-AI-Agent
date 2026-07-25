"""
test_cleaner.py

Unit tests for rag.cleaner.

Responsibilities
----------------
- Verify TextCleaner
- Verify CleanDocument creation
- Verify cleaning statistics
"""

import unittest

from models import (
    CleanDocument,
    Document,
)

from rag.cleaner import TextCleaner


class TestTextCleaner(unittest.TestCase):
    """
    Unit tests for TextCleaner.
    """

    def setUp(self):
        """
        Create a sample document for testing.
        """

        self.cleaner = TextCleaner()

        self.document = Document(
            file_name="sample.txt",
            file_path=None,
            text="""
Skip to main content

BOOK ONE

CHAPTER ONE

SRIMAD BHAGAVATA

This is a sample paragraph.

This is another paragraph.
""",
        )

        self.result = self.cleaner.clean(
            self.document
        )

    # ---------------------------------------------------------
    # Cleaner
    # ---------------------------------------------------------

    def test_returns_clean_document(self):
        """
        Cleaner should return CleanDocument.
        """

        self.assertIsInstance(
            self.result,
            CleanDocument,
        )

    def test_original_document_preserved(self):
        """
        Original document should be retained.
        """

        self.assertIs(
            self.result.document,
            self.document,
        )

    def test_cleaned_text_exists(self):
        """
        Cleaned text should not be empty.
        """

        self.assertTrue(
            len(self.result.cleaned_text) > 0
        )

    def test_character_count_not_increased(self):
        """
        Cleaning should never increase
        the number of characters.
        """

        self.assertLessEqual(
            len(self.result.cleaned_text),
            len(self.document.text),
        )

    def test_statistics_created(self):
        """
        CleanResult should exist.
        """

        self.assertIsNotNone(
            self.result.clean_result
        )

    def test_original_character_count(self):
        """
        Original character count should match.
        """

        self.assertEqual(
            self.result.clean_result.original_characters,
            len(self.document.text),
        )

    def test_removed_character_count(self):
        """
        Removed characters should be >= 0.
        """

        self.assertGreaterEqual(
            self.result.clean_result.removed_characters,
            0,
        )

    def test_processing_time(self):
        """
        Processing time should be recorded.
        """

        self.assertGreater(
            self.result.clean_result.processing_time_seconds,
            0,
        )

    def test_noise_removed(self):
        """
        Noise heading should be removed.
        """

        self.assertNotIn(
            "SRIMAD BHAGAVATA",
            self.result.cleaned_text,
        )


if __name__ == "__main__":

    unittest.main(
        verbosity=2,
    )