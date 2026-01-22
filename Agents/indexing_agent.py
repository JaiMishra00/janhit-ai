"""
Indexing Agent: Stores document chunks with embeddings in Qdrant.
"""

import uuid
from qdrant_client.models import PointStruct
from models.state import GraphState
from config import QDRANT_COLLECTION, QDRANT_URL
from qdrant_client import QdrantClient


def index_documents(state: GraphState) -> GraphState:
    """
    Index document chunks with embeddings into Qdrant.
    
    State inputs:
        - chunks: List of {"doc_id": str, "chunk_id": int, "text": str}
        - chunk_embeddings: List of embedding vectors
        - qdrant_client: Qdrant client instance
        
    Returns:
        Updated state (no new fields added)
    """
    
    chunks = state.get("chunks", [])
    embeddings = state.get("chunk_embeddings", [])
    client = state.get("qdrant_client")
    
    print(f"[INDEXING] Processing {len(chunks)} chunks with {len(embeddings)} embeddings")
    
    if not chunks or not embeddings:
        print("[INDEXING] No chunks or embeddings to index, skipping")
        return state
    
    if len(chunks) != len(embeddings):
        print(f"[INDEXING] ERROR: Mismatch - {len(chunks)} chunks but {len(embeddings)} embeddings")
        return state
    
    if not client:
        print("[INDEXING] ERROR: No Qdrant client provided")
        return state
    
    # Prepare points for Qdrant
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        
        # Create payload that matches your existing schema
        # Use BOTH field names for compatibility
        payload = {
            # New schema fields (for new documents)
            "text": chunk["text"],
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            
            # Old schema fields (for compatibility with existing data)
            "original_text": chunk["text"],
            "source_file": chunk["doc_id"],
            
            # Optional metadata (you can customize these)
            "type": "text",
            "doc_type": "User Upload",
            "category": "General",
            "jurisdiction": "Unknown",
            "authority": "User Provided"
        }
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        points.append(point)
    
    # Upsert to Qdrant
    try:
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        print(f"[INDEXING] Successfully indexed {len(points)} chunks to {QDRANT_COLLECTION}")
    except Exception as e:
        print(f"[INDEXING] ERROR: Failed to index documents: {e}")
        import traceback
        traceback.print_exc()
    
    return state