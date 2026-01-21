"""
Shared state schema for the multi-agent RAG system.
All agents must accept and return this state type.
"""

from typing import List, Dict, Optional, TypedDict


class GraphState(TypedDict):
    """
    Unified state for document processing, embedding, and retrieval pipeline.
    
    Flow:
    1. extraction_agent: files → documents
    2. embedding_agent: documents + query → chunks + embeddings
    3. retrieval_agent: embeddings + filters → matches
    """
    
    # ========== INPUTS ==========
    query: str
    files: Optional[List[str]]  # File paths to PDFs or images
    
    # ========== EXTRACTION OUTPUTS ==========
    documents: List[Dict]  # [{"doc_id": str, "text": str}]
    
    # ========== EMBEDDING OUTPUTS ==========
    standalone_questions: List[str]
    chunks: List[Dict]  # [{"doc_id": str, "chunk_id": int, "text": str}]
    query_embeddings: List[List[float]]
    chunk_embeddings: List[List[float]]
    
    # ========== RETRIEVAL OUTPUTS ==========
    filters: Dict  # {"must": {...}, "should": {...}, "top_k": int}
    matches: List[Dict]  # [{"id": str, "score": float, "payload": {...}}]