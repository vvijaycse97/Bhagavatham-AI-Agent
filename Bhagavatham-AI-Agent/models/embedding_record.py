from dataclasses import dataclass, field


@dataclass(slots=True)
class EmbeddingRecord:
    """
    Represents one embedded chunk.
    """

    chunk_id: str

    source_document: str

    chunk_number: int

    text: str

    character_count: int

    word_count: int

    embedding: list[float]

    metadata: dict = field(default_factory=dict)