import unittest

from models.chunk_statistics import ChunkStatistics


class TestChunkStatistics(unittest.TestCase):
    """
    Unit tests for ChunkStatistics model.
    """

    def setUp(self):

        self.statistics = ChunkStatistics(
            documents_processed=2,
            chunks_created=312,
            total_characters=4477455,
            processing_time_seconds=3.29,
        )

    def test_documents_processed(self):
        """Documents processed should be stored correctly."""

        self.assertEqual(
            self.statistics.documents_processed,
            2,
        )

    def test_chunks_created(self):
        """Chunk count should be stored correctly."""

        self.assertEqual(
            self.statistics.chunks_created,
            312,
        )

    def test_total_characters(self):
        """Total characters should be stored correctly."""

        self.assertEqual(
            self.statistics.total_characters,
            4477455,
        )

    def test_processing_time_seconds(self):
        """Processing time should be stored correctly."""

        self.assertAlmostEqual(
            self.statistics.processing_time_seconds,
            3.29,
            places=2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)