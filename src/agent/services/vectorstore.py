"""ChromaDB vector store operations for arXiv paper chunks."""

import chromadb
from chromadb.config import Settings as ChromaSettings

from agent.config import get_settings
from agent.models import Chunk



_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client (lazy initialization)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str):
    """Get or create a ChromaDB collection by name.

    Returns a chromadb.Collection object (dict OK internally).
    """
    client = _get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection_name: str, chunks: list[Chunk]) -> int:
    """Upsert chunks into ChromaDB collection.

    Args:
        collection_name: Name of the collection
        chunks: List of Chunk objects to upsert

    Returns:
        Number of chunks upserted.
    """
    if not chunks:
        return 0

    collection = get_collection(collection_name)

    # Prepare data for upsert
    ids = [chunk.id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "arxiv_id": chunk.arxiv_id,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "char_start": chunk.char_start,
        }
        for chunk in chunks
    ]

    # Generate embeddings
    from agent.services.chunker import _get_embedding_model
    model = _get_embedding_model()
    embeddings = model.encode(documents, show_progress_bar=False).tolist()

    # Upsert with embeddings
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def query_chunks(
    collection_name: str,
    query_text: str,
    n_results: int = 20,
) -> list[dict]:
    """Query chunks from ChromaDB collection.

    Args:
        collection_name: Name of the collection
        query_text: Query text to search for
        n_results: Number of results to return (default 20)

    Returns:
        List of dicts with keys: chunk_id, text, section, page_start, distance
    """
    collection = get_collection(collection_name)

    # Generate query embedding
    from agent.services.chunker import _get_embedding_model
    model = _get_embedding_model()
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    chunks = []
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            chunks.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "section": metadata.get("section", "Unknown"),
                "page_start": metadata.get("page_start", 0),
                "distance": results["distances"][0][i],
            })

    return chunks


def get_collection_count(collection_name: str) -> int:
    """Get the number of chunks in a collection."""
    try:
        collection = get_collection(collection_name)
        return collection.count()
    except Exception:
        return 0