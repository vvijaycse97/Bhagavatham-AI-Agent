"""
Retriever implementation.
"""

from __future__ import annotations

from rag.base_retriever import BaseRetriever
from rag.query_embedding import QueryEmbedding
from rag.retrieval_result import RetrievalResult
from rag.vector_store import VectorStore


class Retriever(BaseRetriever):
    """
    Retrieves the most relevant document chunks for a user query.
    """

    def __init__(
        self,
        query_embedding: QueryEmbedding,
        vector_store: VectorStore,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            query_embedding:
                Query embedding generator.

            vector_store:
                Vector store implementation.
        """
        self._query_embedding = query_embedding
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant document chunks.

        Args:
            query:
                User query.

            top_k:
                Maximum number of results.

        Returns:
            List of RetrievalResult ordered by relevance.
        """
        if top_k <= 0:
         raise ValueError("top_k must be greater than zero.")
        embedding = self._query_embedding.generate(query)

        search_results = self._vector_store.similarity_search(
            query_embedding=embedding,
            top_k=top_k,
        )

        results: list[RetrievalResult] = []

        for item in search_results:
            results.append(
                RetrievalResult(
                    id=item["id"],
                    text=item["document"],
                    metadata=item.get("metadata", {}),
                    score=item.get("score", 0.0),
                )
            )

        return results