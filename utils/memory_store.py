import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct  # FIXED: Add missing import
from config import MEMORY_COLLECTION


def store_memory(
    client: QdrantClient,
    embedder,
    session_id: str,
    role: str,
    text: str,
    turn_id: int
):
    """
    Store a conversational turn in Qdrant memory collection.
    """
    vector = embedder.embed_query(text)

    point = PointStruct(  # FIXED: Use PointStruct, not models.PointStruct
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "session_id": session_id,
            "turn_id": turn_id,
            "role": role,
            "text": text,
            "timestamp": int(time.time())
        }
    )

    client.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[point]
    )


def retrieve_memory(
    client: QdrantClient,
    embedder,
    session_id: str,
    current_query: str,
    top_k: int = 4
):
    """
    Retrieve relevant conversation history from memory.
    """
    # Embed current query
    query_embedding = embedder.embed_query(current_query)

    # Search memory collection
    res = client.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )

    memories = []
    for p in res.points:
        payload = p.payload or {}
        if payload.get("session_id") == session_id:
            text = payload.get("text")
            if text:
                memories.append(text)

    return memories