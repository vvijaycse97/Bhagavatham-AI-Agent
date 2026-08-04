import shutil
import tempfile
import unittest
import uuid
from unittest.mock import Mock

from models.embedding_record import EmbeddingRecord

from rag.indexer import Indexer
from rag.chroma_vector_store import ChromaVectorStore


class TestVectorIndexPipeline(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.mkdtemp()

        self.collection_name = (
            f"integration_{uuid.uuid4().hex}"
        )

        self.vector_store = ChromaVectorStore(
            persist_directory=self.temp_dir,
            collection_name=self.collection_name,
        )

        self.vector_store.create_collection()

        self.indexer = Indexer(
            vector_store=self.vector_store,
        )


    def tearDown(self):

        try:
            self.vector_store.delete_collection()

        except Exception:
            pass


        shutil.rmtree(
            self.temp_dir,
            ignore_errors=True,
        )


    def test_vector_index_pipeline(self):

       
        embedding_records = [

            EmbeddingRecord(
                chunk_id="chunk_001",
                source_document="bhagavatham.txt",
                chunk_number=1,
                text="Lord Krishna appeared in Mathura.",
                character_count=35,
                word_count=5,
                embedding=[0.1] * 768,
                metadata={
                    "chapter": "1",
                },
            ),

            EmbeddingRecord(
                chunk_id="chunk_002",
                source_document="bhagavatham.txt",
                chunk_number=2,
                text="The devotees glorified the Lord.",
                character_count=33,
                word_count=5,
                embedding=[0.2] * 768,
                metadata={
                    "chapter": "1",
                },
            ),
        ]
        


        indexed_count = self.indexer.index(
            embedding_records
        )


        self.assertEqual(
            indexed_count,
            2,
        )


        stored_count = (
            self.vector_store.count()
        )


        self.assertEqual(
            stored_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()