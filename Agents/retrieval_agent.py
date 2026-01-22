"""
Agent 3: Retrieval Agent
Generates filters and retrieves relevant chunks from Qdrant.
"""

from typing import List
from qdrant_client.models import Filter, FieldCondition, MatchAny

from models.state import GraphState
from config import (
    QDRANT_COLLECTION,
    MEMORY_COLLECTION,
    DEFAULT_TOP_K
)


def retrieve_conversation_memory(
    client,
    embedder,
    session_id: str,
    query: str,
    top_k: int = 3
) -> List[str]:
    """
    Retrieve relevant past conversations for context continuity.
    Uses semantic search on the conversation_memory collection.
    """
    try:
        query_vector = embedder.embed_query(query)

        # Use query_points (new Qdrant API)
        response = client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        memories = []
        for point in response.points:
            payload = point.payload or {}
            # Filter by session
            if payload.get("session_id") == session_id:
                text = payload.get("text", "")
                role = payload.get("role", "unknown")
                if text:
                    memories.append(f"[{role}] {text}")

        return memories
    
    except Exception as e:
        print(f"[MEMORY] Error retrieving conversation memory: {e}")
        return []


def generate_filters(state: GraphState) -> GraphState:
    """
    Generate metadata filters based on query.
    
    State inputs:
        - query: User's question
        - documents: Extracted documents (for deriving filters)
        
    State outputs:
        - filters: Dictionary of metadata filters
        
    Returns:
        Updated state with filters
    """
    
    filters = {
        "must": {},
        "top_k": DEFAULT_TOP_K
    }
    
    # Extract document IDs from extracted documents (if any)
    if state.get("documents"):
        doc_ids = [doc["doc_id"] for doc in state["documents"]]
        if doc_ids:
            filters["must"]["doc_id"] = doc_ids
    
    print(f"[FILTERS] Using document-derived filters: {filters}")
    
    return {
        **state,
        "filters": filters
    }


def retrieve_and_rank(state: GraphState) -> GraphState:
    """
    Retrieve and rank relevant chunks using query embeddings.
    
    State inputs:
        - query_embeddings: List of query vectors
        - filters: Metadata filters
        - embedder: Embedding model (passed at runtime)
        - qdrant_client: Qdrant client (passed at runtime)
        - session_id: Current conversation session
        
    State outputs:
        - matches: List of retrieved chunks with scores
        - relevant_memory: Retrieved conversation history
        
    Returns:
        Updated state with retrieval results
    """
    
    query_embeddings = state.get("query_embeddings", [])
    
    print(f"[RETRIEVAL] Starting retrieval with {len(query_embeddings)} query embeddings")
    
    # Get runtime dependencies
    embedder = state.get("embedder")
    client = state.get("qdrant_client")
    
    if not embedder or not client:
        print("[RETRIEVAL] ERROR: Missing embedder or qdrant_client")
        return {
            **state,
            "matches": [],
            "relevant_memory": []
        }
    
    # Retrieve conversation memory
    relevant_memory = retrieve_conversation_memory(
        client=client,
        embedder=embedder,
        session_id=state.get("session_id", "default"),
        query=state["query"],
        top_k=3
    )
    
    print(f"[RETRIEVAL] Retrieved {len(relevant_memory)} memory items")
    
    if not query_embeddings:
        print("[RETRIEVAL] No query embeddings found")
        return {
            **state,
            "matches": [],
            "relevant_memory": relevant_memory
        }
    
    filters_dict = state.get("filters", {})
    top_k = filters_dict.get("top_k", DEFAULT_TOP_K)
    must_filters = filters_dict.get("must", {})
    
    all_matches = []
    
    # Retrieve for each query embedding
    for idx, query_vector in enumerate(query_embeddings):
        print(f"[RETRIEVAL] Searching with embedding {idx + 1}/{len(query_embeddings)}")
        
        # Build Qdrant filter
        qdrant_filter = None
        if must_filters:
            conditions = []
            for key, value in must_filters.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=value)
                    )
            
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Use query_points (new Qdrant API)
        try:
            response = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True
            )
            
            # Convert to standard format
            for point in response.points:
                all_matches.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
                
        except Exception as e:
            print(f"[RETRIEVAL] Error during search: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Remove duplicates and sort by score
    seen_ids = set()
    unique_matches = []
    
    for match in sorted(all_matches, key=lambda x: x["score"], reverse=True):
        if match["id"] not in seen_ids:
            seen_ids.add(match["id"])
            unique_matches.append(match)
    
    # Limit to top_k
    final_matches = unique_matches[:top_k]
    
    print(f"[RETRIEVAL] Retrieved {len(final_matches)} unique matches")
    
    return {
        **state,
        "matches": final_matches,
        "relevant_memory": relevant_memory
    }