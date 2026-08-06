from datetime import datetime

from rag.corpus_builder import CorpusBuilder
from rag.chunk_pipeline import ChunkPipeline
from rag.embedding_pipeline import EmbeddingPipeline
from rag.embedding_generator import EmbeddingGenerator
from rag.sentence_transformer_provider import SentenceTransformerProvider
from utils.logger import get_logger
from rag.chroma_vector_store import ChromaVectorStore
from rag.indexer import Indexer
from rag.query_embedding import QueryEmbedding
from rag.retriever import Retriever
from rag.retrieval_pipeline import RetrievalPipeline
from config.settings import (
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    VECTOR_COLLECTION_NAME,
)
from utils.performance_profiler import PerformanceProfiler

logger = get_logger(__name__)


def print_banner():

    print("=" * 70)
    print("               Bhagavatham AI Agent")
    print("     Stage 1, Stage 2, Stage 3 , Stage 4  & Stage 5 Pipeline")
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

def print_vector_summary(count):

    print(f"Vectors Stored      : {count}")

def main():
    profiler = PerformanceProfiler()
    start_time = datetime.now()

    logger.info("Starting Bhagavatham AI Agent...")

    print_banner()

    #
    # Stage 1
    #
    profiler.start("Corpus")
    corpus_builder = CorpusBuilder()

    corpus_stats = corpus_builder.build()
    profiler.stop("Corpus")
    print_stage("Stage 1 Summary")

    print_corpus_summary(corpus_stats)

    #
    # Stage 2
    #
    profiler.start("Chunking")
    chunk_pipeline = ChunkPipeline()

    chunk_stats = chunk_pipeline.run(
        corpus_builder.cleaned_documents
    )
    profiler.stop("Chunking")
    profiler.set_item_count(
    "Chunking",
    chunk_stats.chunks_created,
    )
    
    print_stage("Stage 2 Summary")

    print_chunk_summary(chunk_stats)

    #
    # Stage 3
    #
    profiler.start("Embedding")
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
    profiler.stop("Embedding")
    profiler.set_item_count(
    "Embedding",
    embedding_stats.chunks_embedded,
    )
    
    print_stage("Stage 3 Summary")

    print_embedding_summary(
    embedding_stats
    )

    #
    # Stage 4
    #
    profiler.start("Vector DB")
    vector_store = ChromaVectorStore(
        persist_directory=VECTOR_DB_PATH,
        collection_name=VECTOR_COLLECTION_NAME,
    )
   
    vector_store.create_collection()


    indexer = Indexer(
        vector_store=vector_store,
    )


    indexed_count = indexer.index(
        embedding_pipeline.embedding_records
    )
    profiler.stop("Vector DB")
    profiler.set_item_count(
    "Vector DB",
    indexed_count,
    )
    
    print_stage("Stage 4 Summary")

    print_vector_summary(
        indexed_count
    )
    #
    # Stage 5
    #
    profiler.start("Retrieval Engine")

    print("\n" + "=" * 60)
    print("STAGE 5 : RETRIEVAL ENGINE")
    print("=" * 60)


    query_embedding = QueryEmbedding(
        embedding_provider=embedding_provider,
    )

    retriever = Retriever(
        query_embedding=query_embedding,
        vector_store=vector_store,
    )

    retrieval_pipeline = RetrievalPipeline(
        retriever=retriever,
    )
    query = "Who is Prahlada?"
    results = retrieval_pipeline.retrieve(
        query=query,
        top_k=5,
    )
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} result(s)\n")

    for index, result in enumerate(results, start=1):
        print("-" * 60)
        print(f"Result {index}")
        print(f"ID       : {result.id}")
        print(f"Score    : {result.score:.4f}")
        print(f"Metadata : {result.metadata}")
        print(f"Text     :\n{result.text}\n")    

    profiler.stop("Retrieval Engine")
    profiler.set_item_count(
        "Retrieval Engine",
        len(results),
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
    summary = profiler.summary()
    print(summary)
    logger.debug("\n%s", summary)


if __name__ == "__main__":
    main()