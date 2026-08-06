"""
Query embedding generator.

Converts a user query into an embedding vector using the configured
embedding provider.
"""

from __future__ import annotations

from rag.embedding_provider import EmbeddingProvider


class QueryEmbedding:
    """
    Generates an embedding vector for a single user query.
    """

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        """
        Initialize the query embedding generator.

        Args:
            embedding_provider:
                Embedding provider implementation.
        """
        self._embedding_provider = embedding_provider

    def generate(self, query: str) -> list[float]:
        """
        Generate an embedding for a user query.

        Args:
            query:
                User query.

        Returns:
            Embedding vector.

        Raises:
            ValueError:
                If the query is empty or contains only whitespace.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        embeddings = self._embedding_provider.embed([query])

        return embeddings[0]