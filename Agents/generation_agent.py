"""
Agent 4: Response Generation Agent
Generates actionable, cited responses using retrieved context.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Tuple

from models.state import GraphState
from config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL


# ---------------- LLM ----------------

llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="not-needed",
    model=LMSTUDIO_MODEL,
    temperature=0.3
)


# ---------------- PROMPT (UNCHANGED) ----------------

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful legal and policy assistant that provides clear, actionable answers based on available information.

YOUR TASK:
Answer the user's question using the provided context. Be thorough and helpful.

GUIDELINES:
1. Use the provided context as your primary source
2. Cite sources using [Source X] notation when referencing specific information
3. If the context contains relevant information, use it to construct your answer
4. Provide actionable guidance when applicable
5. If the context is incomplete, answer what you can and note what's missing
6. Structure your response clearly with sections if needed
"""),
    ("human", """
Query: {query}

Context:
{context}

Provide a comprehensive answer based on the context above.
""")
])


# ---------------- CONTEXT BUILDER (FIXED) ----------------

def prepare_context(matches: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Prepare context string and citations from retrieved matches.

    CONTRACT (important):
    - matches is a list of dicts
    - each dict has keys: id, score, payload
    - payload is a dict containing: text, doc_id, chunk_id
    """

    if not matches:
        return "", []

    context_parts = []
    citations = []

    for idx, match in enumerate(matches, 1):
        payload = match.get("payload", {})
        score = match.get("score", 0.0)

        # Enforce correct payload shape
        if not isinstance(payload, dict):
            raise ValueError("Invalid payload type; expected dict")

        text = payload.get("text", "").strip()
        if not text:
            continue

        doc_id = payload.get("doc_id", "Unknown")
        chunk_id = payload.get("chunk_id", 0)

        context_parts.append(f"[Source {idx}]\n{text}\n")

        citations.append({
            "source_number": idx,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "score": score,
            "text_preview": text[:200]
        })

    if not context_parts:
        raise ValueError("No valid text found in retrieved payloads")

    return "\n".join(context_parts), citations


# ---------------- RESPONSE GENERATION ----------------

def generate_response(state: GraphState) -> GraphState:
    """
    Generate final response using retrieved context.
    """

    print("[GENERATION] generate_response HIT")

    matches = state.get("matches", [])

    if not matches:
        return {
            **state,
            "context": "",
            "citations": [],
            "final_response": (
                "I couldn't find relevant information in the knowledge base "
                "to answer your question."
            )
        }

    # Build context
    context, citations = prepare_context(matches)

    chain = generation_prompt | llm
    result = chain.invoke({
        "query": state["query"],
        "context": context
    })

    response_text = result.content if hasattr(result, "content") else str(result)

    return {
        **state,
        "context": context,
        "citations": citations,
        "final_response": response_text
    }


# ---------------- OUTPUT FORMATTER (UNCHANGED) ----------------

def format_response_with_metadata(state: GraphState) -> Dict:
    return {
        "query": state.get("query", ""),
        "response": state.get("final_response", ""),
        "sources": state.get("citations", []),
        "metadata": {
            "total_sources": len(state.get("citations", [])),
            "filters_applied": state.get("filters", {}),
            "decomposed_questions": state.get("standalone_questions", [])
        }
    }
