"""
embedding_writer.py

Writes generated embeddings to disk.
"""

from __future__ import annotations

import json
import logging

from config import EMBEDDINGS_DIR
from models import EmbeddingRecord

logger = logging.getLogger(__name__)


class EmbeddingWriter:
    """
    Persists embedding records as JSON.
    """

    def __init__(self) -> None:
        EMBEDDINGS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(
        self,
        source_document: str,
        records: list[EmbeddingRecord],
    ) -> None:
        """
        Write embedding records to disk.
        """

        output_file = (
            EMBEDDINGS_DIR
            / source_document.replace(
                ".txt",
                "_embeddings.json",
            )
        )

        logger.info(
            "Writing %d embedding(s) to %s",
            len(records),
            output_file.name,
        )

        payload = [
            {
                "chunk_id": record.chunk_id,
                "source_document": record.source_document,
                "chunk_number": record.chunk_number,
                "text": record.text,
                "character_count": record.character_count,
                "word_count": record.word_count,
                "embedding": record.embedding,
                "metadata": record.metadata,
            }
            for record in records
        ]

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(
                payload,
                fp,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Embedding file written: %s",
            output_file,
        )