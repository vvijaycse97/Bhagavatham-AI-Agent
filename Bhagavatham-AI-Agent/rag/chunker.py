"""
Chunker

Creates text chunks from cleaned documents.
"""

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from models.chunk import Chunk


class Chunker:
    """
    Creates overlapping chunks from cleaned text.
    """

    def create_chunks(
        self,
        source_document: str,
        text: str,
    ) -> list[Chunk]:
        """
        Split text into overlapping chunks.

        Parameters
        ----------
        source_document : str
            Original document name.

        text : str
            Cleaned text.

        Returns
        -------
        list[Chunk]
        """

        chunks: list[Chunk] = []

        start = 0
        chunk_number = 1

        while start < len(text):

            end = min(
                start + CHUNK_SIZE,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{source_document.rsplit('.', 1)[0]}"
                            f"_{chunk_number:06d}"
                        ),
                        source_document=source_document,
                        chunk_number=chunk_number,
                        text=chunk_text,
                        character_count=len(chunk_text),
                        word_count=len(chunk_text.split()),
                    )
                )

                chunk_number += 1

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

        return chunks