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

    print(f"[FILTERS] Generating filters for query: {state['query']}")

    chain = filter_prompt | llm | filter_parser
    result = chain.invoke({"query": state["query"]})

    must = {
        k: v for k, v in result.dict().items()
        if v is not None
    }
    
    filters = {
        "must": must,
        "top_k": DEFAULT_TOP_K
    }
    
    print(f"[FILTERS] Generated filters: {filters}")
    
    return {
        **state,
        "filters": filters
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

    


    print(f"[RETRIEVAL] Starting retrieval with {len(state.get('query_embeddings', []))} query embeddings")

    results = []
    assert all(isinstance(m, dict) for m in results), "Non-dict match leaked"
    
    # If no query embeddings, return empty matches
    if not state.get("query_embeddings"):
        print("[RETRIEVAL] No query embeddings found")
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
        print(f"[RETRIEVAL] Using filters: {filters['must']}")
    else:
        print("[RETRIEVAL] No filters applied")
    
    # Get top_k from filters or use default
    top_k = filters.get("top_k", DEFAULT_TOP_K)
    
    # Search for each query embedding
    for idx, q_emb in enumerate(state["query_embeddings"]):
        print(f"[RETRIEVAL] Searching with embedding {idx + 1}/{len(state['query_embeddings'])}")
        
        try:
            res = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q_emb,
                limit=top_k,
                with_payload=True,
                query_filter=search_filter
            )

            print(f"[RETRIEVAL] Response type: {type(res)}")
            
            # Version-safe handling
            if isinstance(res, tuple):
                hits = res[0]  # (points, next_page)
                print(f"[RETRIEVAL] Got tuple response with {len(hits)} hits")
            else:
                hits = res  # points only
                print(f"[RETRIEVAL] Got direct response with {len(hits) if hasattr(hits, '__len__') else 'unknown'} hits")
            
            for h_idx, h in enumerate(hits):
                print(f"[RETRIEVAL] Processing hit {h_idx + 1}, type: {type(h)}")
                
                # Handle different response formats
                if isinstance(h, tuple):
                    # Tuple format: (point_id, score, payload) or (point_id, payload)
                    if len(h) == 3:
                        point_id, score, payload = h
                    elif len(h) == 2:
                        point_id, payload = h
                        score = 0.0
                    else:
                        print(f"[RETRIEVAL] Unexpected tuple length: {len(h)}")
                        continue
                else:
                    # Object format (ScoredPoint)
                    point_id = h.id
                    score = getattr(h, 'score', 0.0)
                    payload = h.payload

                print(f"[RETRIEVAL] Hit {h_idx + 1} - ID: {point_id}, Score: {score}, Payload type: {type(payload)}")

                # CRITICAL FIX: Ensure payload is a dict, not wrapped in a list
                if isinstance(payload, list) and len(payload) > 0:
                    print(f"[RETRIEVAL] WARNING: Payload is a list, extracting first element")
                    payload = payload[0]

                results.append({
                    "id": str(point_id),
                    "score": float(score) if score is not None else 0.0,
                    "payload": payload  # Now guaranteed to be a dict
                })

        except Exception as e:
            print(f"[RETRIEVAL] Error searching Qdrant: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Global ranking by score
    results = sorted(
        results,
        key=lambda x: x["score"] if x["score"] is not None else -1,
        reverse=True
    )
    
    print(f"[RETRIEVAL] Total results before deduplication: {len(results)}")
    
    # Deduplicate by ID (keep highest score)
    seen_ids = set()
    unique_results = []
    for r in results:
        if r["id"] not in seen_ids:
            unique_results.append(r)
            seen_ids.add(r["id"])
    
    final_matches = unique_results[:top_k]
    
    print(f"[RETRIEVAL] Final matches: {len(final_matches)}")
    for i, match in enumerate(final_matches[:3]):  # Show first 3
        if isinstance(match, dict):
            score = match["score"]
            payload = match["payload"]
        else:  # ScoredPoint
            score = match.score
            payload = match.payload

        doc_id = payload.get("doc_id", "N/A") if isinstance(payload, dict) else "N/A"

        print(f"[RETRIEVAL]   Match {i+1}: score={score:.4f}, doc_id={doc_id}")
    
    return {
        **state,
        "matches": final_matches
    }