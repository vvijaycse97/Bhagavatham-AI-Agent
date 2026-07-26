"""
Chunk Model

Represents one semantic chunk used by the RAG pipeline.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    """
    Represents a single text chunk.
    """

    chunk_id: str

    source_document: str

    chunk_number: int

    text: str

    character_count: int

    word_count: int

    metadata: dict = field(default_factory=dict)