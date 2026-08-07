"""
Unit tests for RetrievalPipeline.
"""

import unittest

from models.search_result import SearchResult
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
                score=0.10,
            ),
            RetrievalResult(
                id="chunk_002",
                text="Lord Narasimha appeared.",
                metadata={
                    "chapter": 7,
                    "section": 8,
                },
                score=0.20,
            ),
        ]


class TestRetrievalPipeline(unittest.TestCase):
    """Unit tests for RetrievalPipeline."""

    def setUp(self):
        self.retriever = DummyRetriever()

        self.pipeline = RetrievalPipeline(
            self.retriever
        )

    def test_pipeline_calls_retriever(self):
        """Pipeline should delegate retrieval to the retriever."""

        self.pipeline.retrieve(
            "Who is Prahlada?"
        )

        self.assertTrue(
            self.retriever.called
        )

        self.assertEqual(
            self.retriever.received_query,
            "Who is Prahlada?",
        )

    def test_pipeline_passes_top_k(self):
        """Pipeline should pass top_k to the retriever."""

        self.pipeline.retrieve(
            query="Prahlada",
            top_k=2,
        )

        self.assertEqual(
            self.retriever.received_top_k,
            2,
        )

    def test_pipeline_returns_search_results(self):
        """Pipeline should return SearchResult objects."""

        results = self.pipeline.retrieve(
            "Prahlada"
        )

        self.assertEqual(
            len(results),
            2,
        )

        self.assertIsInstance(
            results[0],
            SearchResult,
        )

    def test_pipeline_converts_retrieval_result(self):
        """Pipeline should convert RetrievalResult into SearchResult."""

        results = self.pipeline.retrieve(
            "Prahlada"
        )

        result = results[0]

        self.assertEqual(
            result.chunk_id,
            "chunk_001",
        )

        self.assertEqual(
            result.text,
            "Prahlada was a great devotee.",
        )

        self.assertEqual(
            result.source,
            "",
        )

        self.assertEqual(
            result.metadata["chapter"],
            7,
        )

        self.assertEqual(
            result.metadata["section"],
            2,
        )

        self.assertAlmostEqual(
            result.distance,
            0.10,
            places=6,
        )

    def test_pipeline_calculates_similarity(self):
        """Pipeline should calculate similarity through SearchRanker."""

        results = self.pipeline.retrieve(
            "Prahlada"
        )

        result = results[0]

        expected_similarity = (
            1.0 - (0.10 / 2.0)
        )

        self.assertAlmostEqual(
            result.similarity,
            expected_similarity,
            places=6,
        )

    def test_pipeline_assigns_rank(self):
        """Pipeline should assign ranking positions."""

        results = self.pipeline.retrieve(
            "Prahlada"
        )

        self.assertEqual(
            results[0].rank,
            1,
        )

        self.assertEqual(
            results[1].rank,
            2,
        )

    def test_pipeline_ranks_by_distance(self):
        """Lower distance should receive the better rank."""

        results = self.pipeline.retrieve(
            "Prahlada"
        )

        self.assertEqual(
            results[0].chunk_id,
            "chunk_001",
        )

        self.assertEqual(
            results[1].chunk_id,
            "chunk_002",
        )

        self.assertLess(
            results[0].distance,
            results[1].distance,
        )

    def test_pipeline_applies_similarity_filter(self):
        """Pipeline should remove results below similarity threshold."""

        results = self.pipeline.retrieve(
            "Prahlada",
            similarity_threshold=0.91,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].chunk_id,
            "chunk_001",
        )

    def test_pipeline_applies_top_k(self):
        """Pipeline should limit final results to top_k."""

        results = self.pipeline.retrieve(
            "Prahlada",
            top_k=1,
            similarity_threshold=0.0,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].chunk_id,
            "chunk_001",
        )

    def test_pipeline_empty_results(self):
        """Pipeline should return an empty list when nothing is retrieved."""

        class EmptyRetriever(BaseRetriever):

            def retrieve(
                self,
                query: str,
                top_k: int = 5,
            ) -> list[RetrievalResult]:

                return []

        pipeline = RetrievalPipeline(
            EmptyRetriever()
        )

        results = pipeline.retrieve(
            "Prahlada"
        )

        self.assertEqual(
            results,
            [],
        )

    def test_pipeline_invalid_top_k(self):
        """top_k must be greater than zero."""

        with self.assertRaises(ValueError):
            self.pipeline.retrieve(
                "Prahlada",
                top_k=0,
            )

    def test_pipeline_invalid_similarity_threshold(self):
        """Similarity threshold must be between 0 and 1."""

        with self.assertRaises(ValueError):
            self.pipeline.retrieve(
                "Prahlada",
                similarity_threshold=1.5,
            )


if __name__ == "__main__":
    unittest.main()