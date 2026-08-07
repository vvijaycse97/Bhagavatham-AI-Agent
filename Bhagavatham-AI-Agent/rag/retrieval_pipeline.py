"""
Retrieval pipeline.

Coordinates retrieval, ranking, and filtering.
"""

from __future__ import annotations

from rag.base_retriever import BaseRetriever
from rag.search_filter import SearchFilter
from rag.search_ranker import SearchRanker
from models.search_result import SearchResult


class RetrievalPipeline:
    """
    Coordinates the complete retrieval process.

    Responsibilities:
        - Retrieve raw results
        - Convert results into SearchResult objects
        - Rank search results
        - Filter search results

    The pipeline does not know about any specific vector database.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        ranker: SearchRanker | None = None,
        search_filter: SearchFilter | None = None,
    ) -> None:
        """
        Initialize the retrieval pipeline.

        Args:
            retriever:
                Retriever implementation.

            ranker:
                Search ranking component.

            search_filter:
                Search filtering component.
        """

        self._retriever = retriever

        self._ranker = (
            ranker
            if ranker is not None
            else SearchRanker()
        )

        self._search_filter = (
            search_filter
            if search_filter is not None
            else SearchFilter()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.60,
    ) -> list[SearchResult]:
        """
        Retrieve, rank, and filter relevant document chunks.

        Args:
            query:
                User query.

            top_k:
                Maximum number of final results.

            similarity_threshold:
                Minimum similarity score required.

        Returns:
            Final ranked and filtered SearchResult objects.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        retrieval_results = self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        search_results = self._to_search_results(
            retrieval_results
        )

        ranked_results = self._ranker.rank(
            search_results
        )

        filtered_results = self._search_filter.filter(
            ranked_results,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )

        return filtered_results

    @staticmethod
    def _to_search_results(
        retrieval_results,
    ) -> list[SearchResult]:
        """
        Convert RetrievalResult objects into SearchResult objects.

        The existing RetrievalResult.score currently represents
        the raw vector distance returned by the vector store.
        """

        search_results: list[SearchResult] = []

        for result in retrieval_results:
            source = result.metadata.get(
                "source",
                "",
            )

            search_results.append(
                SearchResult(
                    chunk_id=result.id,
                    text=result.text,
                    source=source,
                    metadata=result.metadata,
                    distance=result.score,
                )
            )

        return search_results