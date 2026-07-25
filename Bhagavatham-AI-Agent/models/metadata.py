from dataclasses import dataclass


@dataclass(slots=True)
class Metadata:
    """
    Metadata associated with a chunk.
    """

    source_file: str

    book: str | None = None

    discourse: str | None = None

    chapter: str | None = None

    speaker: str | None = None

    listener: str | None = None

    topic: str | None = None