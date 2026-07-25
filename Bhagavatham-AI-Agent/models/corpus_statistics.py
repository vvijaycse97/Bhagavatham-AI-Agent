"""
corpus_statistics.py

Aggregate statistics for the processed corpus.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class CorpusStatistics:

    documents_processed: int = 0

    original_characters: int = 0

    cleaned_characters: int = 0

    removed_characters: int = 0

    removed_noise_paragraphs: int = 0

    processing_time_seconds: float = 0.0