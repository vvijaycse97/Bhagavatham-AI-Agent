"""
embedding_pipeline.py

Stage 3 of the Bhagavatham AI pipeline.

Generates embeddings from chunks and writes them to disk.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence

from models import (
    Chunk,
    EmbeddingStatistics,
)

from rag.embedding_generator import EmbeddingGenerator
from rag.embedding_writer import EmbeddingWriter

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Orchestrates embedding generation.
    """

    def __init__(
        self,
        generator: EmbeddingGenerator,
    ) -> None:

        self._generator = generator

        self._writer = EmbeddingWriter()

    def run(
        self,
        chunks: Sequence[Chunk],
    ) -> EmbeddingStatistics:
        """
        Execute the embedding pipeline.
        """

        logger.info("=" * 70)
        logger.info("Starting Embedding Pipeline")
        logger.info("=" * 70)

        start_time = time.perf_counter()

        statistics = EmbeddingStatistics()

        if not chunks:
            logger.info("No chunks supplied.")

            return statistics

        #
        # Group chunks by source document
        #

        grouped_chunks: dict[
            str,
            list[Chunk],
        ] = defaultdict(list)

        for chunk in chunks:
            grouped_chunks[
                chunk.source_document
            ].append(chunk)

        #
        # Process each document
        #

        for (
            source_document,
            document_chunks,
        ) in grouped_chunks.items():

            logger.info(
                "Embedding %s",
                source_document,
            )

            records = self._generator.generate(
                document_chunks
            )

            self._writer.write(
                source_document,
                records,
            )

            statistics.documents_processed += 1

            statistics.chunks_embedded += len(records)

        #
        # Embedding dimension
        #

        statistics.embedding_dimension = (
          self._generator.embedding_dimension()
)

        statistics.processing_time_seconds = (
            time.perf_counter() - start_time
        )

        logger.info(
            "Embedding Pipeline completed successfully."
        )

        logger.info(
            "Documents Processed : %d",
            statistics.documents_processed,
        )

        logger.info(
            "Chunks Embedded     : %d",
            statistics.chunks_embedded,
        )

        logger.info(
            "Embedding Dimension : %d",
            statistics.embedding_dimension,
        )

        return statistics