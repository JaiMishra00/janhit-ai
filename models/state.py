from typing import List, Dict, Optional, TypedDict


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
