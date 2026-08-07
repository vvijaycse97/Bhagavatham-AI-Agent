"""
Unit tests for SearchFilter.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

import unittest

from models.search_result import SearchResult
from rag.search_filter import SearchFilter


class TestSearchFilter(unittest.TestCase):
    """Unit tests for SearchFilter."""

    def setUp(self):
        self.filter = SearchFilter()

    def _create_result(
        self,
        chunk_id: str,
        similarity: float,
        text: str = "Bhagavatham verse",
        rank: int = 1,
    ) -> SearchResult:
        """Create a SearchResult for testing."""

        return SearchResult(
            chunk_id=chunk_id,
            text=text,
            similarity=similarity,
            rank=rank,
            metadata={
                "chapter": 1,
            },
        )

    def test_empty_list_returns_empty_list(self):
        """Filtering an empty list should return an empty list."""

        results = self.filter.filter([])

        self.assertEqual([], results)

    def test_invalid_similarity_threshold_low(self):
        """Similarity threshold below zero should raise ValueError."""

        with self.assertRaises(ValueError):
            self.filter.filter(
                [],
                similarity_threshold=-0.1,
            )

    def test_invalid_similarity_threshold_high(self):
        """Similarity threshold above one should raise ValueError."""

        with self.assertRaises(ValueError):
            self.filter.filter(
                [],
                similarity_threshold=1.1,
            )

    def test_invalid_top_k_zero(self):
        """top_k must be greater than zero."""

        with self.assertRaises(ValueError):
            self.filter.filter(
                [],
                top_k=0,
            )

    def test_invalid_top_k_negative(self):
        """Negative top_k should raise ValueError."""

        with self.assertRaises(ValueError):
            self.filter.filter(
                [],
                top_k=-5,
            )

    def test_remove_empty_results(self):
        """Empty text results should be removed."""

        results = [
            self._create_result(
                "1",
                0.90,
                text="Valid",
            ),
            self._create_result(
                "2",
                0.95,
                text="",
            ),
        ]

        filtered = self.filter.filter(results)

        self.assertEqual(1, len(filtered))
        self.assertEqual("1", filtered[0].chunk_id)

    def test_remove_whitespace_results(self):
        """Whitespace-only results should be removed."""

        results = [
            self._create_result(
                "1",
                0.90,
            ),
            self._create_result(
                "2",
                0.95,
                text="   ",
            ),
        ]

        filtered = self.filter.filter(results)

        self.assertEqual(1, len(filtered))
        self.assertEqual("1", filtered[0].chunk_id)

    def test_remove_duplicate_chunk_ids(self):
        """Duplicate chunk IDs should be removed."""

        results = [
            self._create_result("1", 0.95),
            self._create_result("2", 0.90),
            self._create_result("1", 0.85),
        ]

        filtered = self.filter.filter(
            results,
            similarity_threshold=0.0,
            top_k=10,
        )

        self.assertEqual(2, len(filtered))
        self.assertEqual("1", filtered[0].chunk_id)
        self.assertEqual("2", filtered[1].chunk_id)

    def test_similarity_threshold(self):
        """Results below threshold should be removed."""

        results = [
            self._create_result("1", 0.95),
            self._create_result("2", 0.80),
            self._create_result("3", 0.55),
        ]

        filtered = self.filter.filter(
            results,
            similarity_threshold=0.75,
            top_k=10,
        )

        self.assertEqual(2, len(filtered))

    def test_top_k_limit(self):
        """Only top_k results should be returned."""

        results = [
            self._create_result("1", 0.99),
            self._create_result("2", 0.98),
            self._create_result("3", 0.97),
            self._create_result("4", 0.96),
        ]

        filtered = self.filter.filter(
            results,
            similarity_threshold=0.0,
            top_k=2,
        )

        self.assertEqual(2, len(filtered))
        self.assertEqual("1", filtered[0].chunk_id)
        self.assertEqual("2", filtered[1].chunk_id)

    def test_preserve_order(self):
        """Filtering should preserve ranking order."""

        results = [
            self._create_result("1", 0.99, rank=1),
            self._create_result("2", 0.95, rank=2),
            self._create_result("3", 0.91, rank=3),
        ]

        filtered = self.filter.filter(
            results,
            similarity_threshold=0.0,
            top_k=10,
        )

        self.assertEqual(1, filtered[0].rank)
        self.assertEqual(2, filtered[1].rank)
        self.assertEqual(3, filtered[2].rank)

    def test_preserve_metadata(self):
        """Metadata should remain unchanged."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            similarity=0.95,
            metadata={
                "book": 1,
                "chapter": 3,
            },
        )

        filtered = self.filter.filter(
            [result],
            similarity_threshold=0.0,
        )

        self.assertEqual(
            {
                "book": 1,
                "chapter": 3,
            },
            filtered[0].metadata,
        )

    def test_all_results_filtered(self):
        """Should return empty list when all results are removed."""

        results = [
            self._create_result("1", 0.30),
            self._create_result("2", 0.25),
        ]

        filtered = self.filter.filter(
            results,
            similarity_threshold=0.80,
        )

        self.assertEqual([], filtered)

    def test_no_filtering_required(self):
        """All results should remain when they satisfy all conditions."""

        results = [
            self._create_result("1", 0.99),
            self._create_result("2", 0.98),
            self._create_result("3", 0.97),
        ]

        filtered = self.filter.filter(
            results,
            similarity_threshold=0.90,
            top_k=10,
        )

        self.assertEqual(3, len(filtered))


if __name__ == "__main__":
    unittest.main()