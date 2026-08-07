"""
Unit tests for SearchResult.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

import unittest
from dataclasses import FrozenInstanceError

from models.search_result import SearchResult


class TestSearchResult(unittest.TestCase):
    """Unit tests for SearchResult."""

    def test_create_search_result(self):
        """SearchResult should initialize correctly."""

        result = SearchResult(
            chunk_id="chunk_001",
            text="Sri Krishna is the Supreme Personality.",
        )

        self.assertEqual(
            "chunk_001",
            result.chunk_id,
        )

        self.assertEqual(
            "Sri Krishna is the Supreme Personality.",
            result.text,
        )

    def test_default_values(self):
        """Default values should be assigned correctly."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
        )

        self.assertEqual("", result.source)
        self.assertEqual({}, result.metadata)
        self.assertEqual(0.0, result.distance)
        self.assertEqual(0.0, result.similarity)
        self.assertEqual(0, result.rank)

    def test_has_metadata_true(self):
        """has_metadata() should return True."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            metadata={"chapter": 1},
        )

        self.assertTrue(
            result.has_metadata()
        )

    def test_has_metadata_false(self):
        """has_metadata() should return False."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
        )

        self.assertFalse(
            result.has_metadata()
        )

    def test_get_existing_metadata(self):
        """Should return existing metadata value."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            metadata={
                "chapter": 3,
                "book": 1,
            },
        )

        self.assertEqual(
            3,
            result.get_metadata("chapter"),
        )

    def test_get_missing_metadata_returns_default(self):
        """Missing metadata should return default."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
        )

        self.assertEqual(
            "Unknown",
            result.get_metadata(
                "chapter",
                "Unknown",
            ),
        )

    def test_to_dict(self):
        """to_dict() should return expected dictionary."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            source="Bhagavatham",
            metadata={"chapter": 1},
            distance=0.18,
            similarity=0.84,
            rank=1,
        )

        expected = {
            "chunk_id": "1",
            "text": "Verse",
            "source": "Bhagavatham",
            "metadata": {"chapter": 1},
            "distance": 0.18,
            "similarity": 0.84,
            "rank": 1,
        }

        self.assertEqual(
            expected,
            result.to_dict(),
        )

    def test_score_property(self):
        """score property should return similarity."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
            similarity=0.91,
        )

        self.assertAlmostEqual(
            0.91,
            result.score,
            places=6,
        )

    def test_string_representation(self):
        """String representation should contain useful information."""

        result = SearchResult(
            chunk_id="BG_001",
            text="Verse",
            distance=0.25,
            similarity=0.80,
            rank=1,
        )

        text = str(result)

        self.assertIn(
            "BG_001",
            text,
        )

        self.assertIn(
            "rank=1",
            text,
        )

    def test_search_result_is_immutable(self):
        """SearchResult should be immutable."""

        result = SearchResult(
            chunk_id="1",
            text="Verse",
        )

        with self.assertRaises(FrozenInstanceError):
            result.rank = 2

    def test_search_result_equality(self):
        """Equal objects should compare equal."""

        first = SearchResult(
            chunk_id="1",
            text="Verse",
        )

        second = SearchResult(
            chunk_id="1",
            text="Verse",
        )

        self.assertEqual(
            first,
            second,
        )

    
if __name__ == "__main__":
    unittest.main()