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
    ("system",
    "You are a STRICT JSON generator.\n"
     "Do NOT answer the question.\n"
     "Do NOT explain.\n"
     "Do NOT include assumptions.\n\n"
     "Your task is to generate structured retrieval filters.\n\n"
     "Output format (STRICT):\n"
     "{{\n"
     '  "filters": {{\n'
     '    "doc_type": "<string or null>",\n'
     '    "category": "<string or null>",\n'
     '    "jurisdiction": "<string or null>"\n'
     "  }}\n"
     "}}\n\n"
     "Rules:\n"
     "- If a filter cannot be inferred directly from the query, set it to null\n"
     "- Use terminology exactly as in the query\n"
     "- Output ONLY valid JSON"
    ),
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

    print(type(state))

    chain = filter_prompt | llm | filter_parser
    result = chain.invoke({"query": state["query"]})

    must = {
        k: v for k, v in result.dict().items()
        if v is not None
    }
    
    return {
        **state,
        "filters": {
            "must": must,
            "top_k": DEFAULT_TOP_K
        }
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

    print(type(state))

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
            res = qdrant.query_points(
                QDRANT_COLLECTION,
                q_emb,
                limit=top_k,
                with_payload=True
            )

            # Version-safe handling
            if isinstance(res, tuple):
                hits = res[0]          # (points, next_page)
            else:
                hits = res             # points only
            
            for h in hits:

    # Handle tuple-based hits safely
                if isinstance(h, tuple):
                    if len(h) == 3:
                        point_id, score, payload = h
                    elif len(h) == 2:
                        point_id, payload = h
                        score = 0.0

                    else:
                        continue
                else:
                    # Newer client object style
                    point_id = h.id
                    score = h.score
                    payload = h.payload

                results.append({
                    "id": point_id,
                    "score": score,
                    "payload": payload
                })

        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            continue
    
    # Global ranking by score
    results = sorted(
    results,
    key=lambda x: x["score"] if x["score"] is not None else -1,
    reverse=True
)

    
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