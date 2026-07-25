"""
clean_result.py

Represents the result of the text cleaning process.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class CleanResult:
    """
    Result returned by TextCleaner.
    """

    original_characters: int

    cleaned_characters: int

    removed_characters: int

    removed_noise_paragraphs: int
    
    processing_time_seconds: float