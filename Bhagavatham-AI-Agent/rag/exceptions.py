class EmbeddingException(Exception):
    """Base exception for embedding errors."""


class ModelLoadException(EmbeddingException):
    """Raised when an embedding model fails to load."""


class EmbeddingGenerationException(EmbeddingException):
    """Raised when embedding generation fails."""


class InvalidEmbeddingConfiguration(EmbeddingException):
    """Raised when embedding configuration is invalid."""