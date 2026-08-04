from __future__ import annotations

import logging
from typing import Sequence
from typing import Any

from models.embedding_record import EmbeddingRecord
from rag.vector_store import VectorStore

INDEX_BATCH_SIZE = 5000
logger = logging.getLogger(__name__)


class Indexer:
    """
    Indexes embedding records into the configured vector store.

   Responsibilities:
    - Prepare vector store payloads from embedding records
    - Store embedding records using the configured VectorStore
    """

    def __init__(
        self,
        vector_store: VectorStore,
    ) -> None:

        self._vector_store = vector_store


    def index(
        self,
        embedding_records: Sequence[EmbeddingRecord],
    ) -> int:
        """
        Index embedding records into the vector database.

        Parameters
        ----------
        embedding_records:
         Embedding records to index.

        Returns
        -------
        int
            Number of embedding records indexed.
        """
           
        if not embedding_records:
            logger.info(
                "No embedding records supplied for indexing."
            )
            return 0

        
        logger.info(
                    "Starting indexing for %d embedding record(s).",
                    len(embedding_records),
                )
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []


        for record in embedding_records:

            ids.append(
                record.chunk_id
            )

            embeddings.append(
                record.embedding
            )

            documents.append(
                record.text
            )

            metadata = record.metadata.copy()

            metadata.update(
                {
                    "source_document": record.source_document,
                    "chunk_number": record.chunk_number,
                    "character_count": record.character_count,
                    "word_count": record.word_count,
                }
            )

            metadatas.append(
                metadata
            )

            
        
        for start in range(
            0,
            len(ids),
            INDEX_BATCH_SIZE,
        ):
            end = start + INDEX_BATCH_SIZE
        
            self._vector_store.add_embeddings(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            logger.info(
            "Indexed batch %d-%d",
            start,
            min(end, len(ids)),
            )    
        logger.info(
            "Indexed %d embedding(s).",
            len(ids),
        )
        return len(ids)
     