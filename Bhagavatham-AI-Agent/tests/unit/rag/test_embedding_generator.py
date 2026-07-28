import unittest

from models import Chunk
from unittest.mock import MagicMock
from rag.embedding_generator import EmbeddingGenerator
from rag.embedding_provider import EmbeddingProvider
from rag.exceptions import EmbeddingValidationException


class MockProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embedding_dimension(self) -> int:
        return 3


class InvalidProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Return fewer embeddings than chunks
        return [[0.1, 0.2, 0.3]]

    def embedding_dimension(self) -> int:
        return 3


class TestEmbeddingGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = EmbeddingGenerator(MockProvider())

    def test_empty_input_returns_empty_list(self):
        result = self.generator.generate([])
        self.assertEqual(result, [])

    def test_generate_single_chunk(self):
        chunk = Chunk(
            chunk_id="C1",
            source_document="part1.txt",
            chunk_number=1,
            text="Prahlada was a great devotee.",
            character_count=31,
            word_count=5,
            metadata={"chapter": 7},
        )

        result = self.generator.generate([chunk])

        self.assertEqual(len(result), 1)

        record = result[0]

        self.assertEqual(record.chunk_id, "C1")
        self.assertEqual(record.source_document, "part1.txt")
        self.assertEqual(record.chunk_number, 1)
        self.assertEqual(record.text, "Prahlada was a great devotee.")
        self.assertEqual(record.character_count, 31)
        self.assertEqual(record.word_count, 5)
        self.assertEqual(record.metadata["chapter"], 7)
        self.assertEqual(record.embedding, [0.1, 0.2, 0.3])

    def test_multiple_chunks(self):
        chunks = [
            Chunk(
                chunk_id="C1",
                source_document="part1.txt",
                chunk_number=1,
                text="First chunk",
                character_count=11,
                word_count=2,
                metadata={},
            ),
            Chunk(
                chunk_id="C2",
                source_document="part1.txt",
                chunk_number=2,
                text="Second chunk",
                character_count=12,
                word_count=2,
                metadata={},
            ),
        ]

        result = self.generator.generate(chunks)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].chunk_id, "C1")
        self.assertEqual(result[1].chunk_id, "C2")

    def test_provider_returns_wrong_number_of_embeddings(self):
        generator = EmbeddingGenerator(InvalidProvider())

        chunks = [
            Chunk(
                chunk_id="C1",
                source_document="part1.txt",
                chunk_number=1,
                text="First",
                character_count=5,
                word_count=1,
                metadata={},
            ),
            Chunk(
                chunk_id="C2",
                source_document="part1.txt",
                chunk_number=2,
                text="Second",
                character_count=6,
                word_count=1,
                metadata={},
            ),
        ]

        with self.assertRaises(EmbeddingValidationException):
            generator.generate(chunks)

    def test_provider_called_once(self):
        """
        Provider.embed() should be called exactly once with all chunk texts.
        """

        provider = MagicMock(spec=EmbeddingProvider)

        provider.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        generator = EmbeddingGenerator(provider)

        chunks = [
            Chunk(
                chunk_id="C1",
                source_document="part1.txt",
                chunk_number=1,
                text="First chunk",
                character_count=11,
                word_count=2,
                metadata={},
            ),
            Chunk(
                chunk_id="C2",
                source_document="part1.txt",
                chunk_number=2,
                text="Second chunk",
                character_count=12,
                word_count=2,
                metadata={},
            ),
        ]

        generator.generate(chunks)

        provider.embed.assert_called_once_with(
            [
            "First chunk",
            "Second chunk",
            ]
         )
if __name__ == "__main__":
    unittest.main()