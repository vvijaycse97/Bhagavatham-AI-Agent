import unittest

from rag.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):

    def test_vector_store_is_abstract(self):
        with self.assertRaises(TypeError):
            VectorStore()