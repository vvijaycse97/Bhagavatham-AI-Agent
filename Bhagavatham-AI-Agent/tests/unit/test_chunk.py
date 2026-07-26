import unittest

from models.chunk import Chunk


class TestChunk(unittest.TestCase):
    """
    Unit tests for Chunk model.
    """

    def setUp(self):

        self.chunk = Chunk(
            chunk_id="chunk_0001",
            source_document="part1.txt",
            chunk_number=1,
            text="Prahlada was a great devotee of Lord Vishnu.",
            character_count=46,
            word_count=8,
            metadata={},
        )

    def test_chunk_id(self):
        self.assertEqual(
            self.chunk.chunk_id,
            "chunk_0001",
        )

    def test_source_document(self):
        self.assertEqual(
            self.chunk.source_document,
            "part1.txt",
        )

    def test_chunk_number(self):
        self.assertEqual(
            self.chunk.chunk_number,
            1,
        )

    def test_text(self):
        self.assertIn(
            "Prahlada",
            self.chunk.text,
        )

    def test_character_count(self):
        self.assertEqual(
            self.chunk.character_count,
            46,
        )

    def test_word_count(self):
        self.assertEqual(
            self.chunk.word_count,
            8,
        )
    def test_metadata(self):
        """Metadata should be initialized correctly."""
        self.assertEqual(
        self.chunk.metadata,
        {},
       )

if __name__ == "__main__":
    unittest.main(verbosity=2)