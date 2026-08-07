"""
Integration tests for RetrievalPipeline + PromptBuilder.

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
from rag.prompt_builder import PromptBuilder
from rag.query_embedding import QueryEmbedding
from rag.retrieval_pipeline import RetrievalPipeline
from rag.retriever import Retriever
from rag.sentence_transformer_provider import (
    SentenceTransformerProvider,
)


class TestPromptPipelineIntegration(unittest.TestCase):
    """Integration tests for RetrievalPipeline + PromptBuilder."""

    @classmethod
    def setUpClass(cls):
        """Create the real retrieval pipeline and prompt builder."""

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

        cls.retrieval_pipeline = RetrievalPipeline(
            retriever=retriever,
        )

        cls.prompt_builder = PromptBuilder()

    def test_retrieval_results_can_build_prompt(self):
        """Retrieved SearchResults should build a valid prompt."""

        query = "Who is Prahlada?"

        results = self.retrieval_pipeline.retrieve(
            query=query,
            top_k=5,
        )

        prompt = self.prompt_builder.build(
            query=query,
            results=results,
        )

        self.assertTrue(prompt)

        self.assertIn(
            "CONTEXT",
            prompt,
        )

        self.assertIn(
            "USER QUESTION",
            prompt,
        )

        self.assertIn(
            "Who is Prahlada?",
            prompt,
        )

        self.assertIn(
            "ANSWER",
            prompt,
        )

    def test_retrieval_pipeline_returns_search_results(self):
        """Retrieval pipeline should return SearchResult objects."""

        results = self.retrieval_pipeline.retrieve(
            query="Who is Krishna?",
            top_k=5,
        )

        self.assertGreater(
            len(results),
            0,
        )

        for result in results:
            self.assertIsInstance(
                result,
                SearchResult,
            )

    def test_prompt_contains_retrieved_context(self):
        """Prompt should contain actual retrieved Bhagavatham text."""

        query = "Who is Prahlada?"

        results = self.retrieval_pipeline.retrieve(
            query=query,
            top_k=3,
            similarity_threshold=0.50,
        )

        self.assertGreater(
            len(results),
            0,
        )

        prompt = self.prompt_builder.build(
            query=query,
            results=results,
        )

        for result in results:
            self.assertIn(
            result.text.strip(),
            prompt,
        )

    def test_prompt_contains_sources_when_available(self):
        """Prompt should contain retrieved source information."""

        query = "Who is Narada?"

        results = self.retrieval_pipeline.retrieve(
            query=query,
            top_k=3,
        )

        self.assertGreater(
            len(results),
            0,
        )

        prompt = self.prompt_builder.build(
            query=query,
            results=results,
        )

        sources = [
            result.source.strip()
            for result in results
            if result.source.strip()
        ]

        for source in sources:
            self.assertIn(
                source,
                prompt,
            )


if __name__ == "__main__":
    unittest.main()