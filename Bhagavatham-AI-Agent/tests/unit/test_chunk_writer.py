import json
import unittest

from config.settings import CHUNKS_DIR
from models.chunk import Chunk
from rag.chunk_writer import ChunkWriter


class TestChunkWriter(unittest.TestCase):
    """
    Unit tests for ChunkWriter.
    """

    @classmethod
    def setUpClass(cls):

        cls.writer = ChunkWriter()

        cls.chunks = [
            Chunk(
                chunk_id="part1_000001",
                source_document="part1.txt",
                chunk_number=1,
                text="Prahlada was a great devotee of Lord Vishnu.",
                character_count=46,
                word_count=8,
                metadata={},
            ),
            Chunk(
                chunk_id="part1_000002",
                source_document="part1.txt",
                chunk_number=2,
                text="Narasimha appeared from the pillar.",
                character_count=37,
                word_count=6,
                metadata={},
            ),
        ]

        cls.writer.write(
            "part1.txt",
            cls.chunks,
        )

        cls.json_file = (
            CHUNKS_DIR /
            "part1_chunks.json"
        )

        cls.text_file = (
            CHUNKS_DIR /
            "part1_chunks.txt"
        )

    def test_json_created(self):
        """JSON file should exist."""

        self.assertTrue(
            self.json_file.exists()
        )

    def test_text_created(self):
        """Text file should exist."""

        self.assertTrue(
            self.text_file.exists()
        )

    def test_json_not_empty(self):
        """JSON file should contain data."""

        with open(
            self.json_file,
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        self.assertGreater(
            len(data),
            0,
        )

    def test_text_not_empty(self):
        """Text file should contain data."""

        with open(
            self.text_file,
            encoding="utf-8",
        ) as file:

            content = file.read()

        self.assertGreater(
            len(content),
            0,
        )

    def test_json_contains_chunk_id(self):
        """JSON should contain chunk IDs."""

        with open(
            self.json_file,
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        self.assertEqual(
            data[0]["chunk_id"],
            "part1_000001",
        )

    def test_json_contains_text(self):
        """JSON should contain chunk text."""

        with open(
            self.json_file,
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        self.assertIn(
            "Prahlada",
            data[0]["text"],
        )

    def test_text_contains_chunk_id(self):
        """Text report should contain chunk IDs."""

        with open(
            self.text_file,
            encoding="utf-8",
        ) as file:

            content = file.read()

        self.assertIn(
            "part1_000001",
            content,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)