"""
Agent: Document Indexing
Indexes document chunks into Qdrant vector store.
"""

from qdrant_client import QdrantClient, models
from models.state import GraphState
from config import QDRANT_URL, QDRANT_COLLECTION
import hashlib

# Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_URL)


def generate_point_id(doc_id: str, chunk_id: int) -> str:
    """
    Generate a valid UUID-style point ID from doc_id and chunk_id.
    
    Qdrant requires point IDs to be either:
    - Unsigned integers
    - Valid UUIDs
    
    Args:
        doc_id: Document identifier
        chunk_id: Chunk index
        
    Returns:
        UUID string
    """
    # Create a unique string and hash it to get a UUID
    unique_string = f"{doc_id}::{chunk_id}"
    hash_object = hashlib.md5(unique_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # Format as UUID (8-4-4-4-12)
    uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    
    return uuid_str


def index_documents(state: GraphState) -> GraphState:
    """
    Index document chunks into Qdrant vector store.
    
    State inputs:
        - chunks: List of {"doc_id": str, "chunk_id": int, "text": str}
        - chunk_embeddings: List of embedding vectors
        
    State outputs:
        - (No new state fields, but chunks are indexed in Qdrant)
        
    Returns:
        Updated state (unchanged, indexing is a side effect)
    """
    
    chunks = state.get("chunks", [])
    embeddings = state.get("chunk_embeddings", [])
    
    print(f"[INDEXING] Processing {len(chunks)} chunks with {len(embeddings)} embeddings")
    
    # Skip if no chunks or embeddings
    if not chunks or not embeddings:
        print("[INDEXING] No chunks or embeddings to index, skipping")
        return state
    
    if len(chunks) != len(embeddings):
        print(f"[INDEXING] WARNING: Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        return state
    
    # Prepare points for Qdrant
    points = []
    
    for chunk, embedding in zip(chunks, embeddings):
        doc_id = chunk["doc_id"]
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        
        # Generate valid point ID
        point_id = generate_point_id(doc_id, chunk_id)
        
        # Create point
        point = models.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": text,
            "doc_type": state.get("document_profile", {}).get("doc_type"),
            "category": state.get("document_profile", {}).get("category"),
            "jurisdiction": state.get("document_profile", {}).get("jurisdiction"),
}
        )
        
        points.append(point)
    
    # Upsert to Qdrant
    try:
        print(f"[INDEXING] Upserting {len(points)} points to collection '{QDRANT_COLLECTION}'")
        
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        
        print(f"[INDEXING] Successfully indexed {len(points)} chunks")
        
    except Exception as e:
        print(f"[INDEXING] Error indexing documents: {e}")
        import traceback
        traceback.print_exc()
    
    return state