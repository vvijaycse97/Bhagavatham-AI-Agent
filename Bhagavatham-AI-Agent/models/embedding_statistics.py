"""
Embedding Statistics Model

Stores aggregate statistics for the embedding pipeline.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class EmbeddingStatistics:
    """
    Aggregate statistics produced during embedding generation.
    """

    documents_processed: int = 0

    chunks_embedded: int = 0

    embedding_dimension: int = 0

    processing_time_seconds: float = 0.0