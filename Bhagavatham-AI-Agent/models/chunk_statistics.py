from dataclasses import dataclass


@dataclass(slots=True)
class ChunkStatistics:

    documents_processed: int = 0

    processing_time_seconds: float = 0.0

    chunks_created: int = 0

    total_characters: int = 0

    