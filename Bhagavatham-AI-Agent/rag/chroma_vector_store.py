from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of the VectorStore interface.

    This class manages a single persistent ChromaDB collection.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
    ) -> None:
        self._persist_directory = Path(persist_directory)
        self._collection_name = collection_name

        self._persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing ChromaDB at '%s'",
            self._persist_directory,
        )

        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory)
        )
        logger.info(
         "Persist directory: %s",
          self._persist_directory.resolve(),
        )

        logger.info(
         "Collection name: %s",
          self._collection_name,
        )
        
        self._collection: Collection | None = None

    def create_collection(self, collection_name: str | None = None) -> None:
        """
        Create or load a collection.
        """

        name = collection_name or self._collection_name

        logger.info("Loading collection '%s'", name)

        self._collection = self._client.get_or_create_collection(
            name=name
        )

    def add_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add embeddings to the collection.
        """

        if self._collection is None:
            raise RuntimeError(
                "Collection has not been created."
            )

        logger.info(
            "Adding %d embeddings into collection '%s'",
            len(ids),
            self._collection_name,
        )

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the vector store for the most similar documents.

        Args:
            query_embedding:
                Query embedding vector.

        top_k:
            Maximum number of results to return.

        Returns:
            List of matching documents ordered by similarity.
        """

        if self._collection is None:
            raise RuntimeError(
                "Collection has not been created." 
                " Call create_collection() first."
            )

        logger.info(
            "Performing similarity search (top_k=%d)",
            top_k,
        )

        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = response.get("ids") or [[]]
        documents = response.get("documents") or [[]]
        metadatas = response.get("metadatas") or [[]]
        distances = response.get("distances") or [[]]

        ids = ids[0]
        documents = documents[0]
        metadatas = metadatas[0]
        distances = distances[0]

        results: list[dict[str, Any]] = []

        for id_, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            results.append(
                {
                    "id": id_,
                    "document": document,
                    "metadata": metadata or {},
                    "score": float(distance),
                }
            )

        logger.info(
            "Retrieved %d result(s).",
            len(results),
        )

        return results

    def count(self) -> int:
        """
        Return number of stored vectors.
        """

        if self._collection is None:
            raise RuntimeError(
                "Collection has not been created."
            )

        return self._collection.count()

    def persist(self) -> None:
        """
        ChromaDB automatically persists data.
        """

        logger.info(
            "ChromaDB persistence is automatic."
        )

    def delete_collection(self) -> None:
        """
        Delete the collection.
        """

        logger.warning(
            "Deleting collection '%s'",
            self._collection_name,
        )

        self._client.delete_collection(
            self._collection_name
        )

        self._collection = None