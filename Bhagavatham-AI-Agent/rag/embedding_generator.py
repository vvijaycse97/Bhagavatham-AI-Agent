from typing import List, Dict, Any

from rag.embedding_provider import EmbeddingProvider


class EmbeddingGenerator:
    """Generates embeddings while preserving chunk metadata."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def generate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.provider.embed(texts)

        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": {
                    k: v
                    for k, v in chunk.items()
                    if k not in {"chunk_id", "text"}
                }
            })

        return results