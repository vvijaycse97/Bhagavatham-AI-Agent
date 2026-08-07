"""
Integration tests for RetrievalPipeline.

These tests use the real embedding model and the
existing persistent ChromaDB collection.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

import unittest

from config.settings import (
    EMBEDDING_MODEL,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_PATH,
)
from models.search_result import SearchResult
from rag.chroma_vector_store import ChromaVectorStore
from rag.query_embedding import QueryEmbedding
from rag.retrieval_pipeline import RetrievalPipeline
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
        """Retrieval should return results for a known query."""

        results = self.pipeline.retrieve(
            query="Who is Prahlada?",
            top_k=5,
            similarity_threshold=0.50,
        )

        self.assertGreater(len(results), 0)

    def test_returns_search_result_objects(self):
        """Pipeline should return SearchResult objects."""

        results = self.pipeline.retrieve(
            query="Who is Prahlada?",
            top_k=3,
            similarity_threshold=0.50,
        )

        for result in results:
            self.assertIsInstance(
                result,
                SearchResult,
            )

    def test_top_k_limit_is_respected(self):
        """Pipeline should not return more than top_k results."""

        top_k = 3

        results = self.pipeline.retrieve(
            query="Krishna",
            top_k=top_k,
            similarity_threshold=0.50,
        )

        self.assertLessEqual(
            len(results),
            top_k,
        )

    def test_results_are_ranked(self):
        """Results should have sequential ranks."""

        results = self.pipeline.retrieve(
            query="Prahlada",
            top_k=3,
            similarity_threshold=0.50,
        )

        for expected_rank, result in enumerate(
            results,
            start=1,
        ):
            self.assertEqual(
                expected_rank,
                result.rank,
            )

    def test_results_have_valid_similarity(self):
        """Similarity scores should be between zero and one."""

        results = self.pipeline.retrieve(
            query="Narada",
            top_k=5,
            similarity_threshold=0.50,
        )

        for result in results:
            self.assertGreaterEqual(
                result.similarity,
                0.0,
            )

            self.assertLessEqual(
                result.similarity,
                1.0,
            )

    def test_results_contain_non_empty_text(self):
        """Retrieved results should contain document text."""

        results = self.pipeline.retrieve(
            query="Bhagavatham",
            top_k=5,
            similarity_threshold=0.50,
        )

        for result in results:
            self.assertTrue(
                result.text.strip(),
            )


if __name__ == "__main__":
    unittest.main()