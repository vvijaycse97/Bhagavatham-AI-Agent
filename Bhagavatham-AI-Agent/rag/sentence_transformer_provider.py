"""
Sentence Transformer embedding provider.

This module provides an implementation of the EmbeddingProvider interface
using the sentence-transformers library.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from rag.embedding_provider import EmbeddingProvider
from rag.exceptions import (
    EmbeddingGenerationException,
    ModelLoadException,
)

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using SentenceTransformers.

    The model is loaded lazily on the first embedding request.
    """

    DEFAULT_BATCH_SIZE = 32
    DEFAULT_NORMALIZE_EMBEDDINGS = True

    def __init__(
        self,
        model_name: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
    ) -> None:
        """
        Initialize the embedding provider.

        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformer model.
        batch_size : int, optional
            Batch size used during embedding generation.
        normalize_embeddings : bool, optional
            Whether generated embeddings should be L2 normalized.
        """
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

        self._model: SentenceTransformer | None = None

    def _load_model(self) -> None:
        """
        Load the embedding model if it has not already been loaded.

        Raises
        ------
        ModelLoadException
            If the model cannot be loaded.
        """
        if self._model is not None:
            return

        logger.info("Loading embedding model '%s'...", self._model_name)

        try:
            self._model = SentenceTransformer(self._model_name)

            logger.info(
                "Embedding model '%s' loaded successfully.",
                self._model_name,
            )

        except Exception as exc:
            logger.exception(
                "Failed to load embedding model '%s'.",
                self._model_name,
            )

            raise ModelLoadException(
                f"Unable to load embedding model '{self._model_name}'."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of input texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            Generated embedding vectors.

        Raises
        ------
        EmbeddingGenerationException
            If embedding generation fails.
        """
        if not texts:
            logger.info("No input texts supplied for embedding generation.")
            return []

        self._load_model()

        if self._model is None:
            raise EmbeddingGenerationException(
                "Embedding model is not initialized."
            )

        logger.info(
            "Generating embeddings for %d text(s)...",
            len(texts),
        )

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
            )

            logger.info(
                "Successfully generated %d embedding(s).",
                len(embeddings),
            )

            return embeddings.tolist()

        except Exception as exc:
            logger.exception("Embedding generation failed.")

            raise EmbeddingGenerationException(
                "Failed to generate embeddings."
            ) from exc
