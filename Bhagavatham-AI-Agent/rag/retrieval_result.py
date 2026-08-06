"""
Retrieval result domain model.

Represents a single document chunk retrieved from the vector store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    Represents a single retrieved document chunk.

    Attributes:
        id:
            Unique identifier of the retrieved chunk.

        text:
            Chunk text/content.

        metadata:
            Metadata associated with the chunk
            (chapter, section, source, etc.).

        score:
            Retrieval score returned by the retrieval engine.

            Note:
                This is intentionally generic rather than 'distance'
                because different vector databases expose different
                similarity metrics.
        """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0