"""
Chunk Writer

Writes chunk data to disk.
"""

import json
from dataclasses import asdict

from utils.logger import get_logger
from config.settings import CHUNKS_DIR

from models.chunk import Chunk


logger = get_logger(__name__)


class ChunkWriter:
    """
    Writes chunk files.
    """

    def write(
        self,
        source_document: str,
        chunks: list[Chunk],
    ) -> None:
        """
        Write chunks as JSON and text.
        """

        logger.info("Writing chunk files...")

        file_name = source_document.rsplit(".", 1)[0]

        json_file = CHUNKS_DIR / f"{file_name}_chunks.json"

        text_file = CHUNKS_DIR / f"{file_name}_chunks.txt"

        #
        # JSON
        #

        with open(
            json_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                [asdict(chunk) for chunk in chunks],
                file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "JSON chunk file written: %s",
            json_file,
        )

        #
        # Text
        #

        with open(
            text_file,
            "w",
            encoding="utf-8",
        ) as file:

            for chunk in chunks:

                file.write("=" * 70 + "\n")
                file.write(f"Chunk ID : {chunk.chunk_id}\n")
                file.write(f"Chunk No : {chunk.chunk_number}\n")
                file.write("=" * 70 + "\n\n")

                file.write(chunk.text)

                file.write("\n\n")

        logger.info(
            "Text chunk file written: %s",
            text_file,
        )

        logger.info(
            "Total chunks written: %d",
            len(chunks),
        )