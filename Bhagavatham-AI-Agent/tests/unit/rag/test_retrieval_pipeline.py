"""
Unit tests for RetrievalPipeline.
"""

import unittest

from rag.base_retriever import BaseRetriever
from rag.retrieval_pipeline import RetrievalPipeline
from rag.retrieval_result import RetrievalResult


class DummyRetriever(BaseRetriever):
    """Dummy retriever for testing."""

    def __init__(self):
        self.called = False
        self.received_query = None
        self.received_top_k = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        self.called = True
        self.received_query = query
        self.received_top_k = top_k

        return [
            RetrievalResult(
                id="chunk_001",
                text="Prahlada was a great devotee.",
                metadata={
                    "chapter": 7,
                    "section": 2,
                },
                score=0.98,
            )
        ]


class TestRetrievalPipeline(unittest.TestCase):
    """Unit tests for RetrievalPipeline."""

    def setUp(self):
        self.retriever = DummyRetriever()
        self.pipeline = RetrievalPipeline(self.retriever)

    def test_pipeline_calls_retriever(self):
        """Pipeline should delegate retrieval to the retriever."""

        self.pipeline.retrieve("Who is Prahlada?")

        self.assertTrue(self.retriever.called)
        self.assertEqual(
            self.retriever.received_query,
            "Who is Prahlada?",
        )

    def test_pipeline_passes_top_k(self):
        """Pipeline should pass top_k to the retriever."""

        self.pipeline.retrieve(
            query="Prahlada",
            top_k=10,
        )

        self.assertEqual(
            self.retriever.received_top_k,
            10,
        )

    def test_pipeline_returns_results(self):
        """Pipeline should return RetrievalResult objects."""

        results = self.pipeline.retrieve("Prahlada")

        self.assertEqual(len(results), 1)
        self.assertIsInstance(
            results[0],
            RetrievalResult,
        )

    def test_pipeline_returns_correct_result(self):
        """Pipeline should return the retriever's results unchanged."""

        result = self.pipeline.retrieve("Prahlada")[0]

        self.assertEqual(result.id, "chunk_001")
        self.assertEqual(
            result.text,
            "Prahlada was a great devotee.",
        )
        self.assertEqual(result.score, 0.98)


if __name__ == "__main__":
    unittest.main()