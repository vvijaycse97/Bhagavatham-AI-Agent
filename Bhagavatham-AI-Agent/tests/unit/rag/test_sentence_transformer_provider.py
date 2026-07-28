"""
Unit tests for SentenceTransformerProvider.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from config.settings import EMBEDDING_MODEL

import numpy as np

from rag.exceptions import (
    EmbeddingGenerationException,
    ModelLoadException,
)
from rag.sentence_transformer_provider import (
    SentenceTransformerProvider,
)


class TestSentenceTransformerProvider(unittest.TestCase):
    """Unit tests for SentenceTransformerProvider."""

    def setUp(self) -> None:
        """Create a provider instance for each test."""
        self.provider = SentenceTransformerProvider(
            model_name=EMBEDDING_MODEL,
            batch_size=2,
            normalize_embeddings=True,
        )

    def test_provider_initializes_correctly(self) -> None:
        """Provider should initialize with supplied configuration."""
        self.assertEqual(
            self.provider._model_name,
            EMBEDDING_MODEL,
        )
        self.assertEqual(self.provider._batch_size, 2)
        self.assertTrue(self.provider._normalize_embeddings)
        self.assertIsNone(self.provider._model)

    def test_empty_input_returns_empty_list(self) -> None:
        """Embedding an empty list should return an empty list."""
        result = self.provider.embed([])

        self.assertEqual(result, [])

    @patch("rag.sentence_transformer_provider.SentenceTransformer")
    def test_model_loaded_only_once(
        self,
        mock_sentence_transformer,
    ) -> None:
        """Model should only be loaded once."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        mock_sentence_transformer.return_value = mock_model

        self.provider.embed(["Krishna"])
        self.provider.embed(["Radha"])

        mock_sentence_transformer.assert_called_once()

    def test_provider_uses_configured_model(self):
        """
        Provider should use the configured embedding model.
        """

        self.assertEqual(
        self.provider._model_name,
        EMBEDDING_MODEL,
    )

    @patch("rag.sentence_transformer_provider.SentenceTransformer")
    def test_embed_returns_embeddings(
        self,
        mock_sentence_transformer,
    ) -> None:
        """Embedding generation should return vectors."""
        mock_model = MagicMock()

        expected = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        )

        mock_model.encode.return_value = expected
        mock_sentence_transformer.return_value = mock_model

        result = self.provider.embed(
            [
                "Krishna",
                "Prahlada",
            ]
        )

        self.assertEqual(
            result,
            expected.tolist(),
        )

        mock_model.encode.assert_called_once()

    @patch(
        "rag.sentence_transformer_provider.SentenceTransformer",
        side_effect=Exception("Load failed"),
    )
    def test_model_load_failure_raises_exception(
        self,
        mock_sentence_transformer,
    ) -> None:
        """Model loading errors should raise ModelLoadException."""
        with self.assertRaises(ModelLoadException):
            self.provider.embed(["Krishna"])

    @patch("rag.sentence_transformer_provider.SentenceTransformer")
    def test_embedding_generation_failure_raises_exception(
        self,
        mock_sentence_transformer,
    ) -> None:
        """Embedding failures should raise EmbeddingGenerationException."""
        mock_model = MagicMock()

        mock_model.encode.side_effect = Exception(
            "Embedding failed"
        )

        mock_sentence_transformer.return_value = mock_model

        with self.assertRaises(
            EmbeddingGenerationException
        ):
            self.provider.embed(["Krishna"])

    

if __name__ == "__main__":
    unittest.main()