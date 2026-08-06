"""
Unit tests for RetrievalResult.
"""

import unittest
from dataclasses import FrozenInstanceError

from rag.retrieval_result import RetrievalResult


class TestRetrievalResult(unittest.TestCase):
    """Unit tests for RetrievalResult."""

    def test_create_retrieval_result(self):
        """Should create a RetrievalResult successfully."""

        result = RetrievalResult(
            id="chunk_001",
            text="Prahlada was a great devotee.",
            metadata={
                "chapter": 7,
                "section": 2,
            },
            score=0.98,
        )

        self.assertEqual(result.id, "chunk_001")
        self.assertEqual(
            result.text,
            "Prahlada was a great devotee.",
        )
        self.assertEqual(
            result.metadata,
            {
                "chapter": 7,
                "section": 2,
            },
        )
        self.assertEqual(result.score, 0.98)

    def test_default_metadata(self):
        """Metadata should default to an empty dictionary."""

        result = RetrievalResult(
            id="chunk_001",
            text="Sample text",
        )

        self.assertEqual(result.metadata, {})

    def test_default_score(self):
        """Score should default to 0.0."""

        result = RetrievalResult(
            id="chunk_001",
            text="Sample text",
        )

        self.assertEqual(result.score, 0.0)

    def test_retrieval_result_is_immutable(self):
        """Frozen dataclass should prevent modification."""

        result = RetrievalResult(
            id="chunk_001",
            text="Sample text",
        )

        with self.assertRaises(FrozenInstanceError):
            result.score = 0.95

    def test_two_equal_objects(self):
        """Equal RetrievalResult objects should compare equal."""

        result1 = RetrievalResult(
            id="chunk_001",
            text="Sample text",
            metadata={"chapter": 1},
            score=0.95,
        )

        result2 = RetrievalResult(
            id="chunk_001",
            text="Sample text",
            metadata={"chapter": 1},
            score=0.95,
        )

        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()