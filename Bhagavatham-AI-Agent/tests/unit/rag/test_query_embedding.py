"""
Unit tests for QueryEmbedding.
"""

import unittest

from rag.embedding_provider import EmbeddingProvider
from rag.query_embedding import QueryEmbedding


class DummyEmbeddingProvider(EmbeddingProvider):
    """Dummy embedding provider used for testing."""

    def __init__(self):
        self.embed_called = False
        self.received_texts = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_called = True
        self.received_texts = texts
        return [[0.1, 0.2, 0.3]]


class TestQueryEmbedding(unittest.TestCase):
    """Unit tests for QueryEmbedding."""

    def setUp(self):
        self.provider = DummyEmbeddingProvider()
        self.query_embedding = QueryEmbedding(self.provider)

    def test_generate_embedding_success(self):
        """Should generate an embedding for a valid query."""

        embedding = self.query_embedding.generate(
            "Who is Prahlada?"
        )

        self.assertEqual(
            embedding,
            [0.1, 0.2, 0.3],
        )

    def test_provider_called_once(self):
        """Embedding provider should be called."""

        self.query_embedding.generate(
            "Who is Prahlada?"
        )

        self.assertTrue(self.provider.embed_called)

    def test_provider_receives_single_query_list(self):
        """Provider should receive the query wrapped in a list."""

        query = "Who is Narasimha?"

        self.query_embedding.generate(query)

        self.assertEqual(
            self.provider.received_texts,
            [query],
        )

    def test_empty_query_raises_value_error(self):
        """Empty query should raise ValueError."""

        with self.assertRaises(ValueError):
            self.query_embedding.generate("")

    def test_whitespace_query_raises_value_error(self):
        """Whitespace-only query should raise ValueError."""

        with self.assertRaises(ValueError):
            self.query_embedding.generate("     ")

    def test_query_with_leading_and_trailing_spaces(self):
        """Query with surrounding whitespace should still be accepted."""

        embedding = self.query_embedding.generate(
            "   Prahlada   "
        )

        self.assertEqual(
            embedding,
            [0.1, 0.2, 0.3],
        )


if __name__ == "__main__":
    unittest.main()