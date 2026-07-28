import time

from utils.logger import get_logger

from models.chunk_statistics import ChunkStatistics

from rag.chunker import Chunker
from rag.chunk_writer import ChunkWriter


logger = get_logger(__name__)


class ChunkPipeline:
    """
    Creates chunks from cleaned documents.
    """

    def __init__(self):

        self.chunker = Chunker()

        self.writer = ChunkWriter()
        self.chunks = []

    def run(
        self,
        documents,
    ) -> ChunkStatistics:
        """
        Execute the complete chunk pipeline.
        """

        logger.info("=" * 70)
        logger.info("Starting Chunk Pipeline")
        logger.info("=" * 70)

        start_time = time.perf_counter()

        statistics = ChunkStatistics()

        for clean_document in documents:

            logger.info(
                "Chunking %s",
               clean_document.document.file_name,
            )

            chunks = self.chunker.create_chunks(
                text=clean_document.cleaned_text,
                source_document=clean_document.document.file_name,
            )
            self.chunks.extend(chunks)
            self.writer.write(
            clean_document.document.file_name,
            chunks,
            )

            statistics.documents_processed += 1

            statistics.chunks_created += len(chunks)

            statistics.total_characters += len(
            clean_document.cleaned_text
        )

        statistics.processing_time_seconds = (
            time.perf_counter() - start_time
        )

        logger.info(
            "Chunk Pipeline completed successfully."
        )

        logger.info(
            "Documents Processed : %d",
            statistics.documents_processed,
        )

        logger.info(
            "Chunks Created      : %d",
            statistics.chunks_created,
        )

        return statistics