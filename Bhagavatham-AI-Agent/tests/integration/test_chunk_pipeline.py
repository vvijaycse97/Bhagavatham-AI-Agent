"""
Integration Test

Pipeline:

Raw Files
    ↓
CorpusBuilder
    ↓
Clean Documents
    ↓
ChunkPipeline
    ↓
Chunk Files
"""

import unittest

from config.settings import CHUNKS_DIR

from rag.corpus_builder import CorpusBuilder
from rag.chunk_pipeline import ChunkPipeline


class TestChunkPipeline(unittest.TestCase):
    """
    End-to-end integration test for the chunk pipeline.
    """

    @classmethod
    def setUpClass(cls):

        cls.builder = CorpusBuilder()

        cls.builder.build()

        cls.pipeline = ChunkPipeline()

        cls.statistics = cls.pipeline.run(
            cls.builder.cleaned_documents
        )

    def test_statistics_created(self):
        """Statistics object should be returned."""

        self.assertIsNotNone(self.statistics)

    def test_documents_processed(self):
        """At least one document should be processed."""

        self.assertGreater(
            self.statistics.documents_processed,
            0,
        )

    def test_chunks_created(self):
        """Pipeline should create chunks."""

        self.assertGreater(
            self.statistics.chunks_created,
            0,
        )

    def test_chunks_directory_exists(self):
        """Chunks directory should exist."""

        self.assertTrue(CHUNKS_DIR.exists())

    def test_json_chunk_file_exists(self):
        """JSON chunk files should exist."""

        json_files = list(
            CHUNKS_DIR.glob("*_chunks.json")
        )

        self.assertGreater(
            len(json_files),
            0,
        )

    def test_text_chunk_file_exists(self):
        """Text chunk files should exist."""

        text_files = list(
            CHUNKS_DIR.glob("*_chunks.txt")
        )

        self.assertGreater(
            len(text_files),
            0,
        )

    def test_processing_time(self):
        """Processing time should be recorded."""

        self.assertGreaterEqual(
            self.statistics.processing_time_seconds,
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)