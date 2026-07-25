"""
Unit tests for rag/corpus_builder.py
"""

import unittest

from config import (
    PROCESSED_DATA_DIR,
)

from models import CorpusStatistics

from rag.corpus_builder import CorpusBuilder


class TestCorpusBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.builder = CorpusBuilder()

        cls.statistics = cls.builder.build()

    def test_statistics_instance(self):

        self.assertIsInstance(
            self.statistics,
            CorpusStatistics,
        )

    def test_documents_processed(self):

        self.assertGreater(
            self.statistics.documents_processed,
            0,
        )

    def test_processed_directory_exists(self):

        self.assertTrue(
            PROCESSED_DATA_DIR.exists()
        )

    def test_clean_files_created(self):

        files = list(
            PROCESSED_DATA_DIR.glob("*_clean.txt")
        )

        self.assertGreater(len(files), 0)

    def test_character_counts(self):

        self.assertGreater(
            self.statistics.original_characters,
            self.statistics.cleaned_characters,
        )

    def test_processing_time(self):

        self.assertGreater(
            self.statistics.processing_time_seconds,
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)