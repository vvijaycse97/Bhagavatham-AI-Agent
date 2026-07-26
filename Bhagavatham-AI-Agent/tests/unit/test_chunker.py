import unittest

from rag.chunker import Chunker
from models.chunk import Chunk


class TestChunker(unittest.TestCase):
    """
    Unit tests for Chunker.
    """

    def setUp(self):

        self.chunker = Chunker()

        self.small_text = (
            "Prahlada was a great devotee of Lord Vishnu.\n\n"
            "He remained devoted despite many hardships."
        )

        self.large_text = (
            ("This is a sample paragraph about Bhagavatham.\n\n") * 200
        )

    def test_returns_list(self):
        """Chunker should return a list."""

        chunks = self.chunker.create_chunks(
            "part1.txt",
            self.small_text,
        )

        self.assertIsInstance(chunks, list)

    def test_returns_chunk_objects(self):
        """Every item should be a Chunk."""

        chunks = self.chunker.create_chunks(
            "part1.txt",
            self.small_text,
        )

        self.assertTrue(
            all(isinstance(c, Chunk) for c in chunks)
        )

    def test_single_chunk(self):
        """Small text should produce one chunk."""

        chunks = self.chunker.create_chunks(
            "part1.txt",
            self.small_text,
        )

        self.assertEqual(len(chunks), 1)

    def test_multiple_chunks(self):
        """Large text should produce multiple chunks."""

        chunks = self.chunker.create_chunks(
            "part1.txt",
            self.large_text,
        )

        self.assertGreater(len(chunks), 1)

    def test_chunk_ids_are_unique(self):
        """Chunk IDs should be unique."""

        chunks = self.chunker.create_chunks(
            "part1.txt",
            self.large_text,
        )

        ids = [c.chunk_id for c in chunks]

        self.assertEqual(
            len(ids),
            len(set(ids)),
        )

    def test_positive_word_count(self):
        """Each chunk should have positive word count."""

        chunks = self.chunker.create_chunks(
            "part1.txt",
            self.large_text,
        )

        for chunk in chunks:
            self.assertGreater(
                chunk.word_count,
                0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)