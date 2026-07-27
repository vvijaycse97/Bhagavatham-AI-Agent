"""
Supported embedding model definitions.

This module centralizes the embedding models supported by the
Bhagavatham AI Agent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    """Represents an embedding model."""

    name: str
    dimension: int | None = None


class EmbeddingModels:
    """Supported embedding models."""

    ALL_MINILM_L6_V2 = EmbeddingModel(
        name="all-MiniLM-L6-v2",
        dimension=384,
    )

    # Future models
    BGE_SMALL_EN_V15 = EmbeddingModel(
        name="BAAI/bge-small-en-v1.5",
        dimension=384,
    )

    BGE_BASE_EN_V15 = EmbeddingModel(
        name="BAAI/bge-base-en-v1.5",
        dimension=768,
    )