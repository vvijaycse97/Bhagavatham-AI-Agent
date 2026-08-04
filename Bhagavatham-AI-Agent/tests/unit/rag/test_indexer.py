import unittest
from unittest.mock import Mock

from models.embedding_record import EmbeddingRecord
from rag.indexer import Indexer
from rag.vector_store import VectorStore


class TestIndexer(unittest.TestCase):

    def setUp(self):


        self.vector_store = Mock(
            spec=VectorStore
        )

        self.indexer = Indexer(
            vector_store=self.vector_store,
        )

    def test_empty_embedding_records_returns_zero(self):

        result = self.indexer.index([])

        self.assertEqual(
            result,
            0,
        )

        self.vector_store.add_embeddings.assert_not_called()

    def test_index_single_embedding_record(self):


        embedding_record = EmbeddingRecord(
            chunk_id="chunk_001",
            source_document="bhagavatham.txt",
            chunk_number=1,
            text="Krishna appeared in the world.",
            character_count=35,
            word_count=6,
            embedding=[0.1] * 768,
            metadata={
                "chapter": "1",
            },
        )

        result = self.indexer.index(
            [embedding_record]
        )

        self.assertEqual(
            result,
            1,
        )

        self.vector_store.add_embeddings.assert_called_once_with(
            ids=[
                "chunk_001"
            ],
            embeddings=[
                [0.1] * 768
            ],
            documents=[
                "Krishna appeared in the world."
            ],
            metadatas=[
                {
                    "chapter": "1",
                    "source_document": "bhagavatham.txt",
                    "chunk_number": 1,
                    "character_count": 35,
                    "word_count": 6,
                }
            ],
        )

    def test_index_multiple_embedding_records(self):

        records = [
            EmbeddingRecord(
                chunk_id="chunk_001",
                source_document="book.txt",
                chunk_number=1,
                text="First text",
                character_count=10,
                word_count=2,
                embedding=[0.1] * 768,
                metadata={},
            ),
            EmbeddingRecord(
                chunk_id="chunk_002",
                source_document="book.txt",
                chunk_number=2,
                text="Second text",
                character_count=11,
                word_count=2,
                embedding=[0.2] * 768,
                metadata={},
            ),
        ]

        result = self.indexer.index(records)

        self.assertEqual(
            result,
            2,
        )

        self.vector_store.add_embeddings.assert_called_once()

        call_args = (
            self.vector_store
            .add_embeddings
            .call_args
        )

        self.assertEqual(
            len(call_args.kwargs["ids"]),
            2,
        )

        self.assertEqual(
            len(call_args.kwargs["embeddings"]),
            2,
        )

        self.assertEqual(
            len(call_args.kwargs["documents"]),
            2,
        )

        self.assertEqual(
            len(call_args.kwargs["metadatas"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()