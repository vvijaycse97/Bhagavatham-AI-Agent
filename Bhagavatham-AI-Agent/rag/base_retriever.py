"""
Abstract base class for retrieval engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.retrieval_result import RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval engines.

    Implementations are responsible for retrieving the most relevant
    document chunks for a given user query.
    """

    @abstractmethod
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
                Maximum number of results to return.

        Returns:
            A list of RetrievalResult objects ordered by relevance.
        """
        raise NotImplementedError