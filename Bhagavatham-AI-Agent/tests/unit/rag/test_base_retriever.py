"""
Unit tests for the BaseRetriever abstract class.
"""

import unittest

from rag.base_retriever import BaseRetriever
from rag.retrieval_result import RetrievalResult


class DummyRetriever(BaseRetriever):
    """Concrete implementation used for testing."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        return []


class TestBaseRetriever(unittest.TestCase):
    """Unit tests for BaseRetriever."""

    def test_cannot_instantiate_abstract_class(self):
        """Should not allow instantiation of abstract base class."""

        with self.assertRaises(TypeError):
            BaseRetriever()

    def test_dummy_retriever_returns_empty_list(self):
        """Dummy implementation should return an empty list."""

        retriever = DummyRetriever()

        results = retriever.retrieve("Who is Prahlada?")

        self.assertEqual(results, [])

    def test_dummy_retriever_respects_top_k_parameter(self):
        """Dummy implementation should accept top_k parameter."""

        retriever = DummyRetriever()

        results = retriever.retrieve(
            query="Narasimha",
            top_k=10,
        )

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()