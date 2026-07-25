from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Document:
    """
    Represents a raw source document loaded from disk.
    """

    file_name: str
    file_path: Path
    text: str