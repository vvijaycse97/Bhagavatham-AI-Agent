"""
Search result filtering.

Filters ranked search results before they are passed to the
prompt builder.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

from __future__ import annotations

from models.search_result import SearchResult


class SearchFilter:
    """
    Filters ranked search results.

    Responsibilities:
    - Remove empty results
    - Remove duplicate chunks
    - Filter by similarity threshold
    - Limit number of returned results
    """

    def filter(
        self,
        results: list[SearchResult],
        similarity_threshold: float = 0.60,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Filter ranked search results.

        Args:
            results:
                Ranked search results.

            similarity_threshold:
                Minimum similarity score required.

            top_k:
                Maximum number of results to return.

        Returns:
            Filtered search results.
        """

        self._validate_inputs(
            similarity_threshold,
            top_k,
        )

        filtered = self._remove_empty_results(results)

        filtered = self._remove_duplicates(filtered)

        filtered = self._filter_by_similarity(
            filtered,
            similarity_threshold,
        )

        filtered = self._limit_results(
            filtered,
            top_k,
        )

        return filtered

    def _validate_inputs(
        self,
        similarity_threshold: float,
        top_k: int,
    ) -> None:
        """
        Validate filter parameters.
        """

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

    def _remove_empty_results(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Remove empty search results.
        """

        return [
            result
            for result in results
            if result.text.strip()
        ]

    def _remove_duplicates(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Remove duplicate chunk IDs while preserving order.
        """

        seen: set[str] = set()

        filtered: list[SearchResult] = []

        for result in results:

            if result.chunk_id in seen:
                continue

            seen.add(result.chunk_id)

            filtered.append(result)

        return filtered

    def _filter_by_similarity(
        self,
        results: list[SearchResult],
        threshold: float,
    ) -> list[SearchResult]:
        """
        Keep only results meeting the similarity threshold.
        """

        return [
            result
            for result in results
            if result.similarity >= threshold
        ]

    def _limit_results(
        self,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Limit number of returned results.
        """

        return results[:top_k]