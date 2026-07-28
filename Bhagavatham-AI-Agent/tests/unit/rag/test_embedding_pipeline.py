import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import Chunk
from rag.embedding_pipeline import EmbeddingPipeline


class MockEmbeddingGenerator:
    """Mock embedding generator."""

    def __init__(self):
        self.called = False

    def generate(self, chunks):
        self.called = True

        records = []

        for chunk in chunks:
            records.append(
                type(
                    "EmbeddingRecord",
                    (),
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_document": chunk.source_document,
                        "chunk_number": chunk.chunk_number,
                        "text": chunk.text,
                        "character_count": chunk.character_count,
                        "word_count": chunk.word_count,
                        "embedding": [0.1, 0.2, 0.3],
                        "metadata": chunk.metadata,
                    },
                )()
            )

        return records

    def embedding_dimension(self):
        return 3


class TestEmbeddingPipeline(unittest.TestCase):

    def setUp(self):

        self.generator = MockEmbeddingGenerator()

        self.pipeline = EmbeddingPipeline(
            self.generator
        )

        self.chunk1 = Chunk(
            chunk_id="C1",
            source_document="book1.txt",
            chunk_number=1,
            text="Prahlada",
            character_count=8,
            word_count=1,
            metadata={},
        )

        self.chunk2 = Chunk(
            chunk_id="C2",
            source_document="book1.txt",
            chunk_number=2,
            text="Narasimha",
            character_count=10,
            word_count=1,
            metadata={},
        )

        self.chunk3 = Chunk(
            chunk_id="C3",
            source_document="book2.txt",
            chunk_number=1,
            text="Dhruva",
            character_count=6,
            word_count=1,
            metadata={},
        )

    @patch("rag.embedding_writer.EMBEDDINGS_DIR")
    def test_pipeline_runs_successfully(
        self,
        mock_dir,
    ):
        """
        Pipeline should process all documents.
        """

        with tempfile.TemporaryDirectory() as tmp:

            mock_dir.mkdir.return_value = None

            mock_dir.__truediv__.side_effect = (
                lambda name: Path(tmp) / name
            )

            stats = self.pipeline.run(
                [
                    self.chunk1,
                    self.chunk2,
                    self.chunk3,
                ]
            )

            self.assertEqual(
                stats.documents_processed,
                2,
            )

            self.assertEqual(
                stats.chunks_embedded,
                3,
            )

            self.assertEqual(
                stats.embedding_dimension,
                3,
            )

            self.assertTrue(
                self.generator.called
            )

    def test_empty_input(self):

        stats = self.pipeline.run([])

        self.assertEqual(
            stats.documents_processed,
            0,
        )

        self.assertEqual(
            stats.chunks_embedded,
            0,
        )

    def test_embedding_dimension(self):

        stats = self.pipeline.run(
            [
                self.chunk1,
            ]
        )

        self.assertEqual(
            stats.embedding_dimension,
            3,
        )


if __name__ == "__main__":
    unittest.main()