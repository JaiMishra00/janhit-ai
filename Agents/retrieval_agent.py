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
    Generate Qdrant filters using identified document context.
    """

    doc_profile = state.get("document_profile") or {}

    must = {
        "doc_type": doc_profile.get("doc_type"),
        "category": doc_profile.get("category"),
        "jurisdiction": doc_profile.get("jurisdiction"),
    }

    # Drop null / empty values
    must = {k: v for k, v in must.items() if v}

    filters = {
        "must": must,
        "top_k": DEFAULT_TOP_K
    }

    print(f"[FILTERS] Using document-derived filters: {filters}")

    return {
        **state,
        "filters": filters
    }



def retrieve_and_rank(state: GraphState) -> GraphState:
    """
    Retrieve and rank chunks from Qdrant vector store.
    Compatible with Qdrant returning QueryResponse(points=[...]).
    """

    query_embeddings = state.get("query_embeddings", [])
    print(f"[RETRIEVAL] Starting retrieval with {len(query_embeddings)} query embeddings")

    if not query_embeddings:
        print("[RETRIEVAL] No query embeddings found")
        return {**state, "matches": []}

    results = []

    # -------- Build filter --------
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

    top_k = filters.get("top_k", DEFAULT_TOP_K)

    # -------- Retrieval --------
    for idx, q_emb in enumerate(query_embeddings):
        print(f"[RETRIEVAL] Searching with embedding {idx + 1}/{len(query_embeddings)}")

        try:
            res = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q_emb,
                limit=top_k,
                with_payload=True,
                query_filter=search_filter
            )

            # Qdrant >= 1.7 returns QueryResponse
            points = res.points

            print(f"[RETRIEVAL] Retrieved {len(points)} hits")

            for h in points:
                if not isinstance(h.payload, dict):
                    continue

                payload = h.payload

                # 🔧 BACKWARD-COMPAT FIX
                if "text" not in payload:
                    payload["text"] = payload.get("original_text", "")

                if "doc_id" not in payload:
                    payload["doc_id"] = payload.get("source_file", "Unknown")

                if not payload["text"]:
                    continue

                results.append({
                    "id": str(h.id),
                    "score": float(h.score),
                    "payload": payload
                })

        except Exception as e:
            print(f"[RETRIEVAL] Error searching Qdrant: {e}")
            import traceback
            traceback.print_exc()

    # -------- Rank & Deduplicate --------
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    seen = set()
    unique = []
    for r in results:
        if r["id"] not in seen:
            unique.append(r)
            seen.add(r["id"])

    final_matches = unique[:top_k]

    print(f"[RETRIEVAL] Final matches: {len(final_matches)}")
    for i, m in enumerate(final_matches[:3]):
        print(
            f"[RETRIEVAL]   Match {i+1}: "
            f"score={m['score']:.4f}, "
            f"doc_id={m['payload'].get('doc_id', 'N/A')}"
        )

    return {
        **state,
        "matches": final_matches
    }


    


    