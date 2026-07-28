"""
Unit tests for config/settings.py
"""

import unittest

from config import (
    APP_NAME,
    APP_VERSION,
    AUTHOR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_ENCODING,
    EMBEDDING_MODEL,
    LOG_LEVEL,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
)


class TestConfig(unittest.TestCase):

    def test_app_name(self):
        self.assertEqual(
            APP_NAME,
            "Bhagavatham AI",
        )

    def test_version(self):
        self.assertTrue(APP_VERSION)

    def test_author(self):
        self.assertTrue(AUTHOR)

    def test_chunk_size(self):
        self.assertGreater(
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

    def test_encoding(self):
        self.assertEqual(
            DEFAULT_ENCODING,
            "utf-8",
        )

    def test_embedding_model(self):
        self.assertTrue(
        EMBEDDING_MODEL.startswith("BAAI/")
        )

    def test_log_level(self):
        self.assertEqual(
            LOG_LEVEL,
            "INFO",
        )

    def test_project_root_exists(self):
        self.assertTrue(
            PROJECT_ROOT.exists()
        )

    def test_raw_directory_exists(self):
        self.assertTrue(
            RAW_DATA_DIR.exists()
        )

    def test_reports_directory_exists(self):
        self.assertTrue(
            REPORTS_DIR.exists()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)