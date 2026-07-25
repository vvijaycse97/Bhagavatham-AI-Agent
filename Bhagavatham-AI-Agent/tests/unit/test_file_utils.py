"""
Unit tests for utils/file_utils.py
"""

import unittest

from utils.file_utils import get_clean_filename


class TestFileUtils(unittest.TestCase):
    """Tests for filename helper utilities."""

    def test_txt_file(self):
        """TXT filename should gain _clean suffix."""
        self.assertEqual(
            get_clean_filename("book1.txt"),
            "book1_clean.txt",
        )

    def test_nested_filename(self):
        """Filename with path should still work."""
        self.assertEqual(
            get_clean_filename(
                "folder/book2.txt"
            ),
            "book2_clean.txt",
        )

    def test_pdf_file(self):
        """Extension should be preserved."""
        self.assertEqual(
            get_clean_filename("book.pdf"),
            "book_clean.pdf",
        )

    def test_multiple_dots(self):
        """Filename containing dots should work."""
        self.assertEqual(
            get_clean_filename(
                "part.1.chapter.txt"
            ),
            "part.1.chapter_clean.txt",
        )

    def test_no_extension(self):
        """Filename without extension."""
        self.assertEqual(
            get_clean_filename("book"),
            "book_clean",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)