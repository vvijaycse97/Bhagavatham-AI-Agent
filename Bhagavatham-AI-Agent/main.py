from datetime import datetime

from rag.corpus_builder import CorpusBuilder
from rag.chunk_pipeline import ChunkPipeline
from rag.embedding_pipeline import EmbeddingPipeline
from rag.embedding_generator import EmbeddingGenerator
from rag.sentence_transformer_provider import SentenceTransformerProvider
from config.settings import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


def print_banner():

    print("=" * 70)
    print("               Bhagavatham AI Agent")
    print("     Stage 1, Stage 2 & Stage 3 Pipeline")
    print("=" * 70)


def print_stage(title: str):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_corpus_summary(stats):

    reduction = 0.0

    if stats.original_characters > 0:
        reduction = (
            stats.removed_characters
            / stats.original_characters
        ) * 100

    print(f"Documents Processed      : {stats.documents_processed}")
    print(f"Original Characters      : {stats.original_characters:,}")
    print(f"Cleaned Characters       : {stats.cleaned_characters:,}")
    print(f"Removed Characters       : {stats.removed_characters:,}")
    print(f"Noise Paragraphs Removed : {stats.removed_noise_paragraphs}")
    print(f"Processing Time          : {stats.processing_time_seconds:.2f} sec")
    print(f"Reduction Percentage     : {reduction:.2f}%")



def print_chunk_summary(stats):

    print(f"Documents Processed : {stats.documents_processed}")
    print(f"Chunks Created      : {stats.chunks_created}")
    print(f"Total Characters    : {stats.total_characters:,}")
    print(f"Processing Time     : {stats.processing_time_seconds:.2f} sec")

def print_embedding_summary(stats):

    print(f"Documents Processed : {stats.documents_processed}")
    print(f"Chunks Embedded     : {stats.chunks_embedded}")
    print(f"Embedding Dimension : {stats.embedding_dimension}")
    print(f"Processing Time     : {stats.processing_time_seconds:.2f} sec")


def main():

    start_time = datetime.now()

    logger.info("Starting Bhagavatham AI Agent...")

    print_banner()

    #
    # Stage 1
    #

    corpus_builder = CorpusBuilder()

    corpus_stats = corpus_builder.build()

    print_stage("Stage 1 Summary")

    print_corpus_summary(corpus_stats)

    #
    # Stage 2
    #

    chunk_pipeline = ChunkPipeline()

    chunk_stats = chunk_pipeline.run(
        corpus_builder.cleaned_documents
    )

    print_stage("Stage 2 Summary")

    print_chunk_summary(chunk_stats)

   #
   # Stage 3
   #

    embedding_provider = SentenceTransformerProvider(
      model_name=EMBEDDING_MODEL
    )

    embedding_generator = EmbeddingGenerator(
    embedding_provider
    )

    embedding_pipeline = EmbeddingPipeline(
    embedding_generator
    )

    embedding_stats = embedding_pipeline.run(
    chunk_pipeline.chunks
    )

    print_stage("Stage 3 Summary")

    print_embedding_summary(
    embedding_stats
    )

    #
    # Final Summary
    #

    print()
    print("=" * 70)
    print("Pipeline Completed Successfully")
    print("=" * 70)

    print(f"Execution Time : {datetime.now() - start_time}")

    logger.info("Application completed successfully.")


if __name__ == "__main__":
    main()