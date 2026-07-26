"""
Unit tests for models package.
"""

from importlib.metadata import metadata
import unittest
from pathlib import Path

from models import (
    Chunk,
    CleanDocument,
    CleanResult,
    CorpusStatistics,
    Document,
    Metadata,
)


class TestModels(unittest.TestCase):

    def test_document(self):

        document = Document(
            file_name="book.txt",
            file_path=Path("book.txt"),
            text="Krishna",
        )

        self.assertEqual(
            document.file_name,
            "book.txt",
        )

    def test_clean_result(self):

        result = CleanResult(
            original_characters=100,
            cleaned_characters=90,            
            removed_characters=10,
            removed_noise_paragraphs=1,
            processing_time_seconds=0.5,
        )

        self.assertEqual(
            result.removed_characters,
            10,
        )

    def test_clean_document(self):

        document = Document(
            file_name="a.txt",
            file_path=Path("a.txt"),
            text="text",
        )

        result = CleanResult(
            original_characters=4,
            cleaned_characters=4,        
            removed_characters=0,
            removed_noise_paragraphs=0,
            processing_time_seconds=0,
        )

        clean = CleanDocument(
            document=document,
            cleaned_text="text",
            clean_result=result,
        )

        self.assertEqual(
            clean.cleaned_text,
            "text",
        )

    def test_metadata(self):

        metadata = Metadata(
            source_file="book1.txt"
        )

        self.assertEqual(
            metadata.source_file,
            "book1.txt",
        )
        self.assertIsNone(metadata.book)

    def test_chunk(self):

        metadata = Metadata(
        source_file="book1.txt",
        )

        chunk = Chunk(
          chunk_id="chunk_001",
          source_document="part1.txt",
          chunk_number=1,
          text="Krishna is the Supreme Personality.",
          character_count=36,
          word_count=5,
          metadata=metadata,
        )

        self.assertEqual(
            chunk.chunk_id,
            "chunk_001",
        )

        self.assertEqual(
            chunk.metadata.source_file,
            "book1.txt",
        )

    def test_statistics(self):

        stats = CorpusStatistics()

        self.assertEqual(
            stats.documents_processed,
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)