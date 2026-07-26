import unittest

from models.document import Document
from models.chunk_statistics import ChunkStatistics
from rag.chunk_pipeline import ChunkPipeline
from rag.corpus_builder import CorpusBuilder
from config.settings import CHUNKS_DIR

from rag.chunk_pipeline import ChunkPipeline

class TestChunkPipeline(unittest.TestCase):
    """
    Unit tests for ChunkPipeline.
    """

    @classmethod
    def setUpClass(cls):

        cls.builder = CorpusBuilder()

        cls.builder.build()

        cls.pipeline = ChunkPipeline()

        cls.statistics = cls.pipeline.run(
            cls.builder.cleaned_documents
        )

    def test_statistics_returned(self):
        """Should return ChunkStatistics."""

        self.assertIsInstance(
            self.statistics,
            ChunkStatistics,
        )

    def test_documents_processed(self):
        """Should process one document."""

        self.assertEqual(
            self.statistics.documents_processed,
            1,
        )

    def test_chunks_created(self):
        """Should create at least one chunk."""

        self.assertGreater(
            self.statistics.chunks_created,
            0,
        )

    def test_total_characters(self):
        """Character count should be positive."""

        self.assertGreater(
            self.statistics.total_characters,
            0,
        )

    def test_processing_time(self):
        """Processing time should be positive."""

        self.assertGreaterEqual(
            self.statistics.processing_time_seconds,
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)