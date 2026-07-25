"""
test_cleaning_rules.py

Unit tests for rag.cleaning_rules.

Responsibilities
----------------
- Test individual cleaning functions
- Verify deterministic behavior
- Validate edge cases
"""

import unittest

from rag.cleaning_rules import (
    normalize_text,
    remove_archive_navigation,
    remove_archive_preamble,
    remove_control_characters,
    is_noise_paragraph,
    parse_section_heading,
    remove_noise_paragraphs,
)


class TestCleaningRules(unittest.TestCase):
    """
    Unit tests for cleaning rules.
    """

    # ---------------------------------------------------------
    # normalize_text()
    # ---------------------------------------------------------

    def test_normalize_text(self):
        """
        Multiple spaces and CRLF should be normalized.
        """

        text = "Hello\r\n\r\nWorld"

        result = normalize_text(text)

        self.assertEqual(
            result,
            "Hello\n\nWorld",
        )

    # ---------------------------------------------------------
    # remove_archive_navigation()
    # ---------------------------------------------------------

    def test_remove_archive_navigation(self):
        """
        Archive navigation should be removed.
        """

        text = (
            "Skip to main content\n"
            "Texts\n"
            "Video\n\n"
            "BOOK ONE\n"
            "Actual content"
        )

        result = remove_archive_navigation(text)

        self.assertNotIn(
            "Skip to main content",
            result,
        )

        self.assertIn(
            "BOOK ONE",
            result,
        )

    # ---------------------------------------------------------
    # remove_archive_preamble()
    # ---------------------------------------------------------

    def test_remove_archive_preamble(self):
        """
        Text before BOOK ONE should be removed.
        """

        text = (
            "Random OCR\n"
            "More OCR\n"
            "BOOK ONE\n"
            "Real content"
        )

        result = remove_archive_preamble(text)

        self.assertTrue(
            result.startswith("BOOK ONE")
        )

    # ---------------------------------------------------------
    # remove_control_characters()
    # ---------------------------------------------------------

    def test_remove_control_characters(self):
        """
        Control characters should be removed.
        """

        text = "Hello\x00World\x07"

        result = remove_control_characters(text)

        self.assertEqual(
            result,
            "HelloWorld",
        )

    # ---------------------------------------------------------
    # is_noise_paragraph()
    # ---------------------------------------------------------

    def test_noise_detection(self):
        """
        OCR garbage should be classified as noise.
        """

        paragraph = (
            "SRIMAD BHAGAVATA "
            "ABABA MEME HEREMER"
        )

        self.assertTrue(
            is_noise_paragraph(paragraph)
        )

    def test_valid_paragraph_not_noise(self):
        """
        Real paragraph should not be considered noise.
        """

        paragraph = (
            "Sri Sukadeva said that "
            "King Parikshit listened "
            "carefully."
        )

        self.assertFalse(
            is_noise_paragraph(paragraph)
        )

    # ---------------------------------------------------------
    # parse_section_heading()
    # ---------------------------------------------------------

    def test_parse_book_heading(self):
        """
        BOOK heading should be detected.
        """

        heading = parse_section_heading(
            "BOOK ONE"
        )

        self.assertEqual(
            heading,
            "BOOK ONE",
        )

    def test_parse_chapter_heading(self):
        """
        CHAPTER heading should be detected.
        """

        heading = parse_section_heading(
            "CHAPTER FIVE"
        )

        self.assertEqual(
            heading,
            "CHAPTER FIVE",
        )

    def test_invalid_heading(self):
        """
        Long paragraph should not be treated
        as a heading.
        """

        paragraph = (
            "This is a normal paragraph "
            "that discusses philosophy "
            "and therefore should never "
            "be detected as a heading."
        )

        self.assertIsNone(
            parse_section_heading(paragraph)
        )

    # ---------------------------------------------------------
    # remove_noise_paragraphs()
    # ---------------------------------------------------------

    def test_remove_noise_paragraphs(self):
        """
        Noise paragraphs should be removed.
        """

        text = (
            "BOOK ONE\n\n"
            "ABABA HEREMER MEME\n\n"
            "Sri Sukadeva said..."
        )

        cleaned, removed = remove_noise_paragraphs(
            text
        )

        self.assertEqual(
            removed,
            1,
        )

        self.assertNotIn(
            "ABABA",
            cleaned,
        )

        self.assertIn(
            "Sri Sukadeva",
            cleaned,
        )


if __name__ == "__main__":

    unittest.main(
        verbosity=2,
    )