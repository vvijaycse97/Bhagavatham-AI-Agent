"""
Integration tests for RetrievalPipeline.

These tests use the real embedding model and the
existing persistent ChromaDB collection.
"""

import unittest

from config.settings import (
    EMBEDDING_MODEL,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_PATH,
)

from rag.chroma_vector_store import ChromaVectorStore
from rag.query_embedding import QueryEmbedding
from rag.retrieval_pipeline import RetrievalPipeline
from rag.retrieval_result import RetrievalResult
from rag.retriever import Retriever
from rag.sentence_transformer_provider import (
    SentenceTransformerProvider,
)


class TestRetrievalPipelineIntegration(unittest.TestCase):
    """Integration tests for RetrievalPipeline."""

    @classmethod
    def setUpClass(cls):
        """Create the real retrieval pipeline."""

        embedding_provider = SentenceTransformerProvider(
            model_name=EMBEDDING_MODEL,
        )

        query_embedding = QueryEmbedding(
            embedding_provider=embedding_provider,
        )

        vector_store = ChromaVectorStore(
            persist_directory=VECTOR_DB_PATH,
            collection_name=VECTOR_COLLECTION_NAME,
        )

        vector_store.create_collection()

        retriever = Retriever(
            query_embedding=query_embedding,
            vector_store=vector_store,
        )

        cls.pipeline = RetrievalPipeline(
            retriever=retriever,
        )

    def test_retrieve_returns_results(self):
        """Pipeline should return at least one result."""

        results = self.pipeline.retrieve(
            query="Who is Prahlada?",
            top_k=5,
        )

        self.assertGreater(
            len(results),
            0,
        )

    def test_returns_retrieval_result_objects(self):
        """Returned objects should be RetrievalResult."""

        results = self.pipeline.retrieve(
            query="Who is Prahlada?",
            top_k=5,
        )

        for result in results:
            self.assertIsInstance(
                result,
                RetrievalResult,
            )

    def test_result_contains_expected_data(self):
        """Returned results should contain populated fields."""

        result = self.pipeline.retrieve(
            query="Who is Prahlada?",
            top_k=1,
        )[0]

        self.assertTrue(result.id)

        self.assertTrue(result.text)

        self.assertIsInstance(
            result.metadata,
            dict,
        )

        self.assertIsInstance(
            result.score,
            float,
        )

    def test_top_k_limit_is_respected(self):
        """Pipeline should not return more than top_k."""

        top_k = 3

        results = self.pipeline.retrieve(
            query="Krishna",
            top_k=top_k,
        )

        self.assertLessEqual(
            len(results),
            top_k,
        )

    def test_query_returns_non_empty_text(self):
        """Retrieved text should not be empty."""

        results = self.pipeline.retrieve(
            query="Narada",
            top_k=5,
        )

        for result in results:
            self.assertGreater(
                len(result.text.strip()),
                0,
            )


if __name__ == "__main__":
    unittest.main()