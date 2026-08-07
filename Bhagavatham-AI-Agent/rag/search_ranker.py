"""
Search Ranker

Ranks retrieved SearchResult objects based on vector similarity.

The SearchRanker is intentionally independent of any vector database
implementation. It operates only on SearchResult objects.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

from __future__ import annotations

from models.search_result import SearchResult


class SearchRanker:
    """
    Ranks search results based on vector distance.

    Responsibilities:
        - Calculate similarity scores
        - Sort search results
        - Assign ranks

    Does NOT:
        - Query the vector database
        - Filter results
        - Remove duplicates
        - Build prompts
    """

    @staticmethod
    def calculate_similarity(distance: float) -> float:
        """
        Converts vector distance into a normalized similarity score.

        Formula:
            similarity = 1 / (1 + distance)

        Properties:
            distance = 0.0  -> similarity = 1.0
            distance = 1.0  -> similarity = 0.5
            distance = 2.0  -> similarity = 0.333...

        Args:
            distance:
                Distance returned by the vector database.

        Returns:
            Similarity score between 0 and 1.
        """
        if distance < 0:
            raise ValueError("Distance cannot be negative.")

        return 1.0 / (1.0 + distance)

    def rank(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Rank search results by ascending distance.

        Similarity scores and ranking positions are recalculated.

        Args:
            results:
                List of SearchResult objects.

        Returns:
            Ranked list of SearchResult objects.
        """
        if not results:
            return []

        # Sort by smallest distance first
        sorted_results = sorted(results, key=lambda result: result.distance)

        ranked_results: list[SearchResult] = []

        for index, result in enumerate(sorted_results, start=1):
            similarity = self.calculate_similarity(result.distance)

            ranked_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    source=result.source,
                    metadata=result.metadata,
                    distance=result.distance,
                    similarity=similarity,
                    rank=index,
                )
            )

        return ranked_results