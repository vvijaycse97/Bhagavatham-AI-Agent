import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
import uuid

from rag.chroma_vector_store import ChromaVectorStore
from rag.vector_store import VectorStore


class TestChromaVectorStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.collection_name = (
            f"test_{uuid.uuid4().hex}"
        )

        self.store = ChromaVectorStore(
            persist_directory=self.temp_dir,
            collection_name=self.collection_name,
        )

        self.store.create_collection()

    def tearDown(self):
        try:
            self.store.delete_collection()
        except Exception:
            pass

        shutil.rmtree(
            self.temp_dir,
            ignore_errors=True,
        )

    def test_is_vector_store(self):
        self.assertIsInstance(
            self.store,
            VectorStore,
        )

    def test_create_collection(self):
        self.assertIsNotNone(
            self.store._collection
        )

    def test_empty_collection_count(self):
        self.assertEqual(
            self.store.count(),
            0,
        )

    def test_add_single_embedding(self):
        self.store.add_embeddings(
            ids=["1"],
            embeddings=[[0.1] * 768],
            documents=["Hello Bhagavatham"],
            metadatas=[
                {
                    "source": "unit_test",
                    "chapter": "1",
                }
            ],
        )

        self.assertEqual(
            self.store.count(),
            1,
        )

    def test_add_multiple_embeddings(self):
        ids = [
            "1",
            "2",
            "3",
        ]

        embeddings = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ]

        documents = [
            "Document 1",
            "Document 2",
            "Document 3",
        ]

        metadatas = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
        ]

        self.store.add_embeddings(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        self.assertEqual(
            self.store.count(),
            3,
        )

    def test_delete_collection(self):
        self.store.delete_collection()

        with self.assertRaises(RuntimeError):
            self.store.count()
    

    def test_similarity_search_returns_results(self):
        """Similarity search should return mapped search results."""

        self.store._collection = MagicMock()

        self.store._collection.query.return_value = {
            "ids": [["chunk1"]],
            "documents": [["Sri Krishna is the Supreme Personality"]],
            "metadatas": [[{"chapter": 1}]],
            "distances": [[0.18]],
        }

        results = self.store.similarity_search(
            [0.1] * 768,
            top_k=5,
        )

        self.assertEqual(1, len(results))

        self.assertEqual("chunk1", results[0]["id"])
        self.assertEqual(
            "Sri Krishna is the Supreme Personality",
            results[0]["document"],
        )
        self.assertEqual(
            {"chapter": 1},
            results[0]["metadata"],
        )
        self.assertAlmostEqual(
            0.18,
            results[0]["score"],
            places=6,
        )

        self.store._collection.query.assert_called_once_with(
            query_embeddings=[[0.1] * 768],
            n_results=5,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    def test_similarity_search_without_collection_raises_runtime_error(self):
        """Searching before creating a collection should fail."""

        self.store._collection = None

        with self.assertRaises(RuntimeError):
            self.store.similarity_search(
                [0.1] * 768
            )

if __name__ == "__main__":
    unittest.main()