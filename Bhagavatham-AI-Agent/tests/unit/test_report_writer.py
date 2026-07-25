"""
Unit tests for rag/report_writer.py
"""

import json
import unittest

from config import (
    CORPUS_REPORT_JSON,
    CORPUS_REPORT_TXT,
)

from models import CorpusStatistics

from rag.report_writer import ReportWriter


class TestReportWriter(unittest.TestCase):

    def setUp(self):

        self.writer = ReportWriter()

        self.statistics = CorpusStatistics(
            documents_processed=2,
            original_characters=1000,
            cleaned_characters=850,
            removed_characters=150,
            removed_noise_paragraphs=8,
            processing_time_seconds=1.25,
        )

        self.writer.write(self.statistics)

    def test_text_report_created(self):
        """Text report should exist."""
        self.assertTrue(CORPUS_REPORT_TXT.exists())

    def test_json_report_created(self):
        """JSON report should exist."""
        self.assertTrue(CORPUS_REPORT_JSON.exists())

    def test_json_contents(self):
        """JSON values should match statistics."""

        data = json.loads(
            CORPUS_REPORT_JSON.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            data["documents_processed"],
            2,
        )

        self.assertEqual(
            data["removed_characters"],
            150,
        )

    def test_text_contains_title(self):

        report = CORPUS_REPORT_TXT.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "Bhagavatham AI Corpus Report",
            report,
        )

    def test_text_contains_document_count(self):

        report = CORPUS_REPORT_TXT.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "Documents Processed",
            report,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)