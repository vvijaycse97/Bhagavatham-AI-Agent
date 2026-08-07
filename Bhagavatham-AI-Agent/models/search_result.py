"""
Search Result Model

Represents a single search result returned by the retrieval engine.

This model is intentionally independent of any vector database implementation
(e.g., ChromaDB) so that the retrieval pipeline can remain database-agnostic.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Represents a single retrieved search result.

    Attributes:
        chunk_id:
            Unique identifier of the retrieved chunk.

        text:
            Text content of the retrieved chunk.

        source:
            Source document or corpus name.

        metadata:
            Additional metadata associated with the chunk.

        distance:
            Raw distance returned by the vector database.
            Lower values indicate a closer match.

        similarity:
            Normalized similarity score.
            Higher values indicate a better match.

        rank:
            Ranking position after search result ranking.
            Rank starts from 1.
    """

    chunk_id: str
    text: str

    source: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    distance: float = 0.0
    similarity: float = 0.0

    rank: int = 0

    def has_metadata(self) -> bool:
        """
        Returns True if metadata is available.

        Returns:
            bool
        """
        return bool(self.metadata)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieves a metadata value.

        Args:
            key:
                Metadata key.

            default:
                Value returned if the key is absent.

        Returns:
            Metadata value or default.
        """
        return self.metadata.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the search result into a dictionary.

        Returns:
            Dictionary representation of the search result.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "distance": self.distance,
            "similarity": self.similarity,
            "rank": self.rank,
        }

    @property
    def score(self) -> float:
        """
        Alias for similarity score.

        Returns:
            Similarity score.
        """
        return self.similarity

    def __str__(self) -> str:
        """
        Human-readable representation.

        Returns:
            String representation.
        """
        return (
            f"SearchResult("
            f"rank={self.rank}, "
            f"similarity={self.similarity:.4f}, "
            f"distance={self.distance:.4f}, "
            f"chunk_id='{self.chunk_id}')"
        )