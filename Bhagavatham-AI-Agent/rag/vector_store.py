from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Abstract interface for vector database implementations."""

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """Create or load a collection."""

    @abstractmethod
    def add_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store embeddings and metadata."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar documents."""

    @abstractmethod
    def count(self) -> int:
        """Return number of vectors stored."""

    @abstractmethod
    def persist(self) -> None:
        """Persist database to disk if required."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection."""