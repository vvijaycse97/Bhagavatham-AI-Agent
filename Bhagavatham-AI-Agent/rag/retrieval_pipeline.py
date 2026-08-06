"""
Retrieval pipeline.

Coordinates the retrieval process using the configured retriever.
"""

from __future__ import annotations

from rag.base_retriever import BaseRetriever
from rag.retrieval_result import RetrievalResult


class RetrievalPipeline:
    """
    Coordinates the retrieval process.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
    ) -> None:
        """
        Initialize the retrieval pipeline.

        Args:
            retriever:
                Retriever implementation.
        """
        self._retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant document chunks.

        Args:
            query:
                User query.

            top_k:
                Maximum number of results.

        Returns:
            List of retrieved results.
        """
        return self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )