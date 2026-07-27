from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract contract for all embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts."""
        raise NotImplementedError

   