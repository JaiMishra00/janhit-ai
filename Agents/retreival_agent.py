"""
Agent 3: Retrieval Agent
Generates metadata filters and retrieves relevant chunks from Qdrant.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from qdrant_client import QdrantClient, models

from models.state import GraphState
from models.schemas import RetrieverFilter
from config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
    QDRANT_URL,
    QDRANT_COLLECTION,
    DEFAULT_TOP_K
)

# Initialize LLM for filter generation
llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="not-needed",
    model=LMSTUDIO_MODEL,
    temperature=0.0
)

# Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_URL)

# Filter generation parser
filter_parser = PydanticOutputParser(pydantic_object=RetrieverFilter)

# Filter generation prompt
filter_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Extract payload filters for vector retrieval.

Rules:
- Only infer filters if clearly implied
- Use metadata fields (only): doc_type, category, year_enacted, jurisdiction, issuing_authority, status, domain
- Do NOT hallucinate values
- Output valid JSON only
"""),
    ("human", "{query}")
])


def generate_filters(state: GraphState) -> GraphState:
    """
    Generate metadata filters from user query.
    
    State inputs:
        - query: Original user question
        
    State outputs:
        - filters: {"must": {...}, "should": {...}, "top_k": int}
        
    Returns:
        Updated state with retrieval filters
    """
    chain = filter_prompt | llm | filter_parser
    result = chain.invoke({"query": state["query"]})
    
    return {
        **state,
        "filters": result.dict()
    }


def retrieve_and_rank(state: GraphState) -> GraphState:
    """
    Retrieve and rank chunks from Qdrant vector store.
    
    State inputs:
        - query_embeddings: List of query vectors
        - filters: Metadata filters
        
    State outputs:
        - matches: List of {"id": str, "score": float, "payload": {...}}
        
    Returns:
        Updated state with retrieved matches
    """
    results = []
    
    # If no query embeddings, return empty matches
    if not state.get("query_embeddings"):
        return {
            **state,
            "matches": []
        }
    
    # Build Qdrant filter
    search_filter = None
    filters = state.get("filters", {})
    
    if filters.get("must"):
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v)
                )
                for k, v in filters["must"].items()
            ]
        )
    
    # Get top_k from filters or use default
    top_k = filters.get("top_k", DEFAULT_TOP_K)
    
    # Search for each query embedding
    for q_emb in state["query_embeddings"]:
        try:
            hits = qdrant.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=q_emb,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True
            )
            
            for h in hits:
                results.append({
                    "id": h.id,
                    "score": h.score,
                    "payload": h.payload
                })
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            continue
    
    # Global ranking by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    # Deduplicate by ID (keep highest score)
    seen_ids = set()
    unique_results = []
    for r in results:
        if r["id"] not in seen_ids:
            unique_results.append(r)
            seen_ids.add(r["id"])
    
    return {
        **state,
        "matches": unique_results[:top_k]  # Limit to top_k globally
    }