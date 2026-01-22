import time
import uuid
from qdrant_client import models


def store_memory(
    client,
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

    point = models.PointStruct(
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
        collection_name="conversation_memory",
        points=[point]
    )


def retrieve_memory(
    client,
    embedder,
    session_id: str,
    current_query: str,
    top_k: int = 4
):
    """
    Retrieve relevant past conversation turns for a session.
    """
    query_vec = embedder.embed_query(current_query)

    hits = client.search(
        collection_name="conversation_memory",
        query_vector=query_vec,
        limit=top_k,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id)
                )
            ]
        )
    )

    return [hit.payload["text"] for hit in hits]
