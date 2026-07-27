import unittest

from rag.embedding_generator import EmbeddingGenerator
from rag.embedding_provider import EmbeddingProvider


class MockProvider(EmbeddingProvider):
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embedding_dimension(self):
        return 3


class TestEmbeddingGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = EmbeddingGenerator(MockProvider())

    def test_generate_embeddings(self):
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Prahlada was a great devotee.",
                "chapter": 7
            }
        ]

        result = self.generator.generate(chunks)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chunk_id"], "C1")
        self.assertEqual(result[0]["metadata"]["chapter"], 7)
        self.assertEqual(len(result[0]["embedding"]), 3)

    def test_empty_input(self):
        result = self.generator.generate([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()