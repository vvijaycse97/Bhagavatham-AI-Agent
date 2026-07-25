"""
Integration Test

Pipeline:

Raw Files
    ↓
Loader
    ↓
Cleaner
    ↓
Processed Files
    ↓
Reports
"""

import unittest

from config.settings import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

from rag.corpus_builder import CorpusBuilder


class TestCorpusBuilderPipeline(unittest.TestCase):
    """
    End-to-end integration test for the corpus builder.
    """

    @classmethod
    def setUpClass(cls):
        cls.builder = CorpusBuilder()

        # Run the complete pipeline
        cls.statistics = cls.builder.build()

    def test_statistics_created(self):
        """Statistics object should be returned."""
        self.assertIsNotNone(self.statistics)

    def test_documents_processed(self):
        """At least one document should be processed."""
        self.assertGreater(
            self.statistics.documents_processed,
            0,
        )

    def test_processed_directory_exists(self):
        """Processed directory should exist."""
        self.assertTrue(PROCESSED_DATA_DIR.exists())

    def test_clean_files_exist(self):
        """Clean text files should be created."""
        clean_files = list(
            PROCESSED_DATA_DIR.glob("*_clean.txt")
        )

        self.assertGreater(len(clean_files), 0)

    def test_report_directory_exists(self):
        """Reports directory should exist."""
        self.assertTrue(REPORTS_DIR.exists())

    def test_text_report_exists(self):
        """Text report should exist."""
        self.assertTrue(
            (REPORTS_DIR / "corpus_report.txt").exists()
        )

    def test_json_report_exists(self):
        """JSON report should exist."""
        self.assertTrue(
            (REPORTS_DIR / "corpus_report.json").exists()
        )

    def test_statistics_consistency(self):
        """Processed documents should equal clean files."""

        clean_files = list(
        PROCESSED_DATA_DIR.glob("*_clean.txt")
        )

        self.assertEqual(
            self.statistics.documents_processed,
            len(clean_files),
        )

    def test_reports_not_empty(self):
        """Reports should contain data."""

        txt = REPORTS_DIR / "corpus_report.txt"

        self.assertGreater(
            txt.stat().st_size,
            0,
        )    
    
if __name__ == "__main__":
    unittest.main(verbosity=2)