"""
Pydantic models for structured LLM outputs.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel


class Topic(BaseModel):
    """Single decomposed query topic."""
    question: str


class DecomposedQuery(BaseModel):
    """Output of query decomposition."""
    topics: List[Topic]


class RetrieverFilter(BaseModel):
    """Filters for vector search with Qdrant."""
    doc_type: Optional[str] = None
    category: Optional[str] = None
    jurisdiction: Optional[str] = None