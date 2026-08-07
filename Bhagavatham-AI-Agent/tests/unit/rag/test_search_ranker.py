"""
Unit tests for SearchRanker.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

import unittest
from dataclasses import FrozenInstanceError

from models.search_result import SearchResult
from rag.search_ranker import SearchRanker


class TestSearchRanker(unittest.TestCase):
    """Unit tests for SearchRanker."""

    def setUp(self):
        """Create SearchRanker instance for each test."""
        self.ranker = SearchRanker()

    def test_rank_empty_list_returns_empty_list(self):
        """Ranking an empty list should return an empty list."""

        ranked = self.ranker.rank([])

        self.assertEqual([], ranked)

    def test_single_result_gets_rank_one(self):
        """A single result should receive rank 1."""

        result = SearchResult(
            chunk_id="1",
            text="Bhagavatham",
            distance=0.20,
        )

        ranked = self.ranker.rank([result])

        self.assertEqual(1, len(ranked))
        self.assertEqual(1, ranked[0].rank)

    def test_results_are_sorted_by_distance(self):
        """Results should be sorted by ascending distance."""

        results = [
            SearchResult(chunk_id="3", text="Verse 3", distance=0.42),
            SearchResult(chunk_id="1", text="Verse 1", distance=0.18),
            SearchResult(chunk_id="2", text="Verse 2", distance=0.29),
        ]

        ranked = self.ranker.rank(results)

        self.assertEqual("1", ranked[0].chunk_id)
        self.assertEqual("2", ranked[1].chunk_id)
        self.assertEqual("3", ranked[2].chunk_id)

    def test_rank_values_are_assigned_correctly(self):
        """Ranks should be sequential."""

        results = [
            SearchResult(chunk_id="3", text="Verse 3", distance=0.42),
            SearchResult(chunk_id="1", text="Verse 1", distance=0.18),
            SearchResult(chunk_id="2", text="Verse 2", distance=0.29),
        ]

        ranked = self.ranker.rank(results)

        self.assertEqual(1, ranked[0].rank)
        self.assertEqual(2, ranked[1].rank)
        self.assertEqual(3, ranked[2].rank)

    def test_similarity_is_calculated_correctly(self):
        """Similarity calculation should be correct."""

        result = SearchResult(
            chunk_id="1",
            text="Bhagavatham",
            distance=0.50,
        )

        ranked = self.ranker.rank([result])

        expected = 1.0 / (1.0 + 0.50)

        self.assertAlmostEqual(
            expected,
            ranked[0].similarity,
            places=6,
        )

    def test_negative_distance_raises_value_error(self):
        """Negative distance should raise ValueError."""

        with self.assertRaises(ValueError):
            self.ranker.calculate_similarity(-0.5)

    def test_original_list_is_not_modified(self):
        """Original list should remain unchanged."""

        results = [
            SearchResult(chunk_id="2", text="Verse 2", distance=0.50),
            SearchResult(chunk_id="1", text="Verse 1", distance=0.10),
        ]

        original = list(results)

        self.ranker.rank(results)

        self.assertEqual(original, results)

    def test_metadata_is_preserved(self):
        """Metadata should be preserved."""

        metadata = {
            "book": 1,
            "chapter": 3,
            "verse": "1.3.28",
        }

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            metadata=metadata,
            distance=0.20,
        )

        ranked = self.ranker.rank([result])

        self.assertEqual(metadata, ranked[0].metadata)

    def test_source_is_preserved(self):
        """Source should be preserved."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            source="Srimad Bhagavatham",
            distance=0.20,
        )

        ranked = self.ranker.rank([result])

        self.assertEqual(
            "Srimad Bhagavatham",
            ranked[0].source,
        )

    def test_distance_is_preserved(self):
        """Distance should not change."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            distance=0.37,
        )

        ranked = self.ranker.rank([result])

        self.assertAlmostEqual(
            0.37,
            ranked[0].distance,
            places=6,
        )

    def test_calculate_similarity_zero_distance(self):
        """Zero distance should return similarity of 1."""

        similarity = self.ranker.calculate_similarity(0.0)

        self.assertAlmostEqual(
            1.0,
            similarity,
            places=6,
        )

    def test_calculate_similarity_large_distance(self):
        """Large distances should produce lower similarity."""

        similarity = self.ranker.calculate_similarity(10.0)

        self.assertAlmostEqual(
            1.0 / 11.0,
            similarity,
            places=6,
        )

    def test_rank_returns_new_objects(self):
        """Ranking should create new SearchResult instances."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            distance=0.25,
        )

        ranked = self.ranker.rank([result])

        self.assertIsNot(result, ranked[0])

    def test_ranked_objects_remain_immutable(self):
        """Returned SearchResult objects should remain immutable."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            distance=0.20,
        )

        ranked = self.ranker.rank([result])

        with self.assertRaises(FrozenInstanceError):
            ranked[0].rank = 5