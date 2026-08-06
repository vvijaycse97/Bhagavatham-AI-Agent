"""
Unit tests for Retriever.
"""

import unittest
from typing import Any

from rag.query_embedding import QueryEmbedding
from rag.retrieval_result import RetrievalResult
from rag.retriever import Retriever
from rag.vector_store import VectorStore


class DummyQueryEmbedding(QueryEmbedding):
    """Dummy query embedding generator."""

    def __init__(self):
        self.called = False
        self.received_query = None

    def generate(self, query: str) -> list[float]:
        self.called = True
        self.received_query = query
        return [0.1, 0.2, 0.3]


class DummyVectorStore(VectorStore):
    """Dummy vector store."""

    def __init__(self):
        self.received_embedding = None
        self.received_top_k = None

    def create_collection(self, collection_name: str) -> None:
        pass

    def add_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        pass

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        self.received_embedding = query_embedding
        self.received_top_k = top_k

        return [
            {
                "id": "chunk_001",
                "document": "Prahlada was a great devotee.",
                "metadata": {
                    "chapter": 7,
                    "section": 2,
                },
                "score": 0.98,
            },
            {
                "id": "chunk_002",
                "document": "Lord Narasimha appeared.",
                "metadata": {
                    "chapter": 7,
                    "section": 8,
                },
                "score": 0.95,
            },
        ]

    def count(self) -> int:
        return 2

    def persist(self) -> None:
        pass

    def delete_collection(self) -> None:
        pass


class EmptyVectorStore(DummyVectorStore):
    """Dummy vector store returning no results."""

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        self.received_embedding = query_embedding
        self.received_top_k = top_k

        return []


class TestRetriever(unittest.TestCase):
    """Unit tests for Retriever."""

    def setUp(self):
        self.query_embedding = DummyQueryEmbedding()
        self.vector_store = DummyVectorStore()

        self.retriever = Retriever(
            query_embedding=self.query_embedding,
            vector_store=self.vector_store,
        )

    def test_generate_query_embedding_called(self):
        """Retriever should generate query embedding."""

        self.retriever.retrieve("Who is Prahlada?")

        self.assertTrue(self.query_embedding.called)
        self.assertEqual(
            self.query_embedding.received_query,
            "Who is Prahlada?",
        )

    def test_similarity_search_called(self):
        """Retriever should call vector store."""

        self.retriever.retrieve(
            "Prahlada",
            top_k=3,
        )

        self.assertEqual(
            self.vector_store.received_embedding,
            [0.1, 0.2, 0.3],
        )

        self.assertEqual(
            self.vector_store.received_top_k,
            3,
        )

    def test_returns_retrieval_results(self):
        """Retriever should return RetrievalResult objects."""

        results = self.retriever.retrieve("Prahlada")

        self.assertEqual(len(results), 2)

        self.assertIsInstance(
            results[0],
            RetrievalResult,
        )

    def test_result_values(self):
        """Retriever should correctly populate RetrievalResult."""

        result = self.retriever.retrieve("Prahlada")[0]

        self.assertEqual(result.id, "chunk_001")
        self.assertEqual(
            result.text,
            "Prahlada was a great devotee.",
        )
        self.assertEqual(
            result.metadata["chapter"],
            7,
        )
        self.assertEqual(
            result.score,
            0.98,
        )

    def test_empty_results(self):
        """Retriever should return an empty list."""

        retriever = Retriever(
            query_embedding=self.query_embedding,
            vector_store=EmptyVectorStore(),
        )

        results = retriever.retrieve("Prahlada")

        self.assertEqual(results, [])

    def test_invalid_top_k(self):
        """top_k must be greater than zero."""

        with self.assertRaises(ValueError):
            self.retriever.retrieve(
                "Prahlada",
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()