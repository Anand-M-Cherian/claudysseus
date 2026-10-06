import os
from functools import lru_cache
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PayloadSchemaType


from claudysseus.config import config
from claudysseus.context.indexers.code_parser import parse_file, get_source_files
from claudysseus.llm.factory import get_embedder
from claudysseus.observability.logger import get_logger


logger = get_logger(__name__)


def index_codebase(repo_path: str) -> QdrantVectorStore:
    """
    Parse all source files in repo_path, embed each chunk and store in Qdrant.
    Returns the QdrantVectorStore. Skips indexing if collection already has data.
    """
    embedder = get_embedder()
    collection_name = config["qdrant"]["collection_name"]
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    # Check if collection already has data
    client = QdrantClient(url=url, api_key=api_key)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        info = client.get_collection(collection_name)
        if info.points_count > 0:
            logger.info(
                f"Loaded existing index with {info.points_count} chunks")
            return QdrantVectorStore.from_existing_collection(
                embedding=embedder,
                url=url,
                api_key=api_key,
                collection_name=collection_name,
            )

    logger.info(f"Starting semantic indexing of {repo_path}")
    docs = []
    for filepath in get_source_files(repo_path):
        docs.extend(_file_documents(filepath))

    vector_store = QdrantVectorStore.from_documents(
        docs, embedder,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        batch_size=50,
    )

    logger.info(f"Semantic indexing complete. Total chunks: {len(docs)}")
    return vector_store


def _file_documents(filepath: str) -> list[Document]:
    """Parse one file into Documents. Returns [] on parse error."""
    try:
        chunks = parse_file(filepath)
    except (SyntaxError, ValueError) as e:
        logger.error(f"Skipping {filepath}: {e}")
        return []

    return [
        Document(
            page_content=chunk.content,
            metadata={
                "source": chunk.source,
                "name": chunk.name,
                "type": chunk.type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
        )
        for chunk in chunks
    ]


@lru_cache(maxsize=1)
def _get_vector_store() -> QdrantVectorStore:
    """Connect to the existing collection once; reused by every watcher event."""
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=get_embedder(),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=config["qdrant"]["collection_name"],
    )
    # Qdrant (strict mode) refuses to filter on unindexed payload fields,
    # and _delete_file_points filters on metadata.source. Idempotent.
    vector_store.client.create_payload_index(
        collection_name=config["qdrant"]["collection_name"],
        field_name="metadata.source",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    return vector_store


def _delete_file_points(vector_store: QdrantVectorStore, filepath: str) -> None:
    """Delete every point whose metadata.source is filepath."""
    vector_store.client.delete(
        collection_name=config["qdrant"]["collection_name"],
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="metadata.source",
                           match=MatchValue(value=filepath))
        ])),
    )


# ------------------------------------------------------------------
# Public API used by watcher.py (per-file, real-time)
# ------------------------------------------------------------------


def index_single_file(filepath: str) -> None:
    """Replace all chunks for filepath with freshly parsed ones."""
    if not os.path.exists(filepath):
        # File may have been deleted between the event firing and this call
        return
    vector_store = _get_vector_store()
    _delete_file_points(vector_store, filepath)
    docs = _file_documents(filepath)
    if docs:
        vector_store.add_documents(docs)
    logger.info(f"Watchdog indexed {filepath} → {len(docs)} chunks")


def remove_file_from_index(filepath: str) -> None:
    """Remove all chunks for a deleted file."""
    _delete_file_points(_get_vector_store(), filepath)
    logger.info(f"Watchdog removed {filepath} from index")


def show_index(vector_store: QdrantVectorStore) -> None:
    """Display all documents stored in the Qdrant collection."""
    from rich.console import Console
    console = Console()
    client = vector_store.client
    collection_name = config["qdrant"]["collection_name"]
    results = client.scroll(collection_name=collection_name,
                            with_payload=True, with_vectors=True, limit=1000)
    points = results[0]
    console.print(f"\n[bold]Semantic Index — {len(points)} chunks[/bold]\n")

    for i, point in enumerate(points):
        payload = point.payload
        emb = point.vector
        console.print(
            f"[bold cyan]── Chunk {i+1} ──────────────────────────[/bold cyan]")
        console.print(f"  File     : {payload['source']}")
        console.print(f"  Name     : {payload['name']} ({payload['type']})")
        console.print(
            f"  Lines    : {payload['start_line']} - {payload['end_line']}")
        console.print(
            f"  Embedding: [{', '.join(f'{v:.4f}' for v in emb[:5])}...] ({len(emb)} dims)")
        console.print(
            f"  Code     :\n[dim]{payload.get('page_content', '')[:300]}[/dim]\n")
