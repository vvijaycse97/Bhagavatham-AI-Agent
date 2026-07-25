from dataclasses import dataclass

from models.metadata import Metadata


@dataclass(slots=True)
class Chunk:
    """
    Represents a semantic chunk ready for embedding.
    """

    chunk_id: str

    text: str

    metadata: Metadata