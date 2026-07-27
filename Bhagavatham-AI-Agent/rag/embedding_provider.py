from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract contract for all embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""
        pass