"""
embedding_generator.py

Generates embeddings for chunked Bhagavatham documents.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from models import Chunk, EmbeddingRecord
from rag.embedding_provider import EmbeddingProvider
from rag.exceptions import EmbeddingValidationException

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for Chunk objects using an EmbeddingProvider.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
    ) -> None:
        self._provider = provider

    def generate(
        self,
        chunks: Sequence[Chunk],
    ) -> list[EmbeddingRecord]:
        """
        Generate embeddings for chunks.

        Parameters
        ----------
        chunks : Sequence[Chunk]
            Chunks to embed.

        Returns
        -------
        list[EmbeddingRecord]
            Embedded chunk records.
        """

        if not chunks:
            logger.info(
                "No chunks supplied for embedding generation."
            )
            return []

        logger.info(
            "Generating embeddings for %d chunk(s).",
            len(chunks),
        )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self._provider.embed(texts)

        if len(embeddings) != len(chunks):
            raise EmbeddingValidationException(
                "Embedding provider returned an unexpected number of embeddings."
            )

        records: list[EmbeddingRecord] = []

        for chunk, embedding in zip(chunks, embeddings):

            records.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    source_document=chunk.source_document,
                    chunk_number=chunk.chunk_number,
                    text=chunk.text,
                    character_count=chunk.character_count,
                    word_count=chunk.word_count,
                    embedding=embedding,
                    metadata=chunk.metadata.copy(),
                )
            )
            

        logger.info(
            "Generated %d embedding record(s).",
            len(records),
        )

        return records

    def embedding_dimension(self) -> int:
        return self._provider.embedding_dimension()