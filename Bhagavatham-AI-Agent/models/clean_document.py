"""
clean_document.py

Represents a cleaned Bhagavatham document.
"""

from dataclasses import dataclass

from .document import Document
from .clean_result import CleanResult


@dataclass(slots=True)
class CleanDocument:
    """
    Output of the TextCleaner.
    """

    document: Document

    cleaned_text: str

    clean_result: CleanResult