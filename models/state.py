from typing import List, Dict, Optional, TypedDict, Any


class GraphState(TypedDict, total=False):
    """
    Unified state for document processing, embedding, and retrieval pipeline.
    """

    # ========== INPUTS ==========
    query: str
    files: Optional[List[str]]

    # ========== EXTRACTION OUTPUTS ==========
    documents: List[Dict]

    # ========== EMBEDDING OUTPUTS ==========
    standalone_questions: List[str]
    chunks: List[Dict]
    query_embeddings: List[List[float]]
    chunk_embeddings: List[List[float]]

    # ========== RETRIEVAL OUTPUTS ==========
    filters: Dict
    matches: List[Dict]

    # ========== GENERATION OUTPUTS ==========
    context: str
    citations: List[Dict]
    final_response: str

    # Legacy fields (if still used elsewhere)
    user_query: str
    retrieved_docs: List[dict]
    final_answer: str

    # ========== CONVERSATIONAL MEMORY ==========
    session_id: str
    relevant_memory: List[str]
    
    # ========== RUNTIME DEPENDENCIES ==========
    # These are passed at runtime and not persisted
    embedder: Any  # The embedding model instance
    qdrant_client: Any  # The Qdrant client instance