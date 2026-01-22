"""
Pydantic schemas for structured outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Topic(BaseModel):
    """Single decomposed topic/question."""
    question: str = Field(description="Standalone question for this topic")


class DecomposedQuery(BaseModel):
    """Result of query decomposition."""
    topics: List[Topic] = Field(description="List of standalone questions")


class FilterSchema(BaseModel):
    """Metadata filters for retrieval."""
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters to apply during retrieval"
    )
    
    class Config:
        # Allow arbitrary field names in filters dict
        extra = "allow"


# Alias for backward compatibility
RetrieverFilter = FilterSchema