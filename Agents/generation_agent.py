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


# ---------------- PROMPT ----------------

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


# ---------------- CONTEXT BUILDER ----------------

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
            print(f"[CONTEXT] Warning: Invalid payload type for match {idx}: {type(payload)}")
            continue

        # Try multiple possible text field names
        text = (
            payload.get("text") or 
            payload.get("original_text") or 
            payload.get("content") or 
            ""
        ).strip()
        
        if not text:
            print(f"[CONTEXT] Warning: No text in payload for match {idx}. Payload keys: {list(payload.keys())}")
            continue

        # Try multiple possible ID field names
        doc_id = (
            payload.get("doc_id") or 
            payload.get("source_file") or 
            payload.get("source") or 
            "Unknown"
        )
        
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
        print(f"[CONTEXT] ERROR: No valid text found in {len(matches)} matches")
        print(f"[CONTEXT] First match payload: {matches[0].get('payload') if matches else 'N/A'}")
        # Return empty instead of raising - let the generation handle it
        return "", []

    return "\n".join(context_parts), citations


# ---------------- MEMORY BLOCK BUILDER ----------------

def build_memory_block(memory_chunks: List[str]) -> str:
    """
    Formats conversational memory for prompt injection.
    Memory is for conversational continuity only and is NOT authoritative.
    """
    if not memory_chunks:
        return ""

    formatted = "\n".join(f"- {m}" for m in memory_chunks)

    return f"""
Conversation Context (for continuity only):
The following reflects earlier parts of this conversation.
It may help interpret follow-up questions but MUST NOT be treated as legal fact.

{formatted}
"""


# ---------------- RESPONSE GENERATION ----------------

def generate_response(state: GraphState) -> GraphState:
    """
    Generate final response using retrieved context.
    """

    print("[GENERATION] generate_response HIT")

    matches = state.get("matches", [])
    relevant_memory = state.get("relevant_memory", [])

    if not matches:
        print("[GENERATION] No matches found - returning fallback response")
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

    # Check if context building failed
    if not context:
        print("[GENERATION] Context preparation failed - returning fallback response")
        return {
            **state,
            "context": "",
            "citations": [],
            "final_response": (
                "I found some matches but couldn't extract readable text from them. "
                "This might be a data formatting issue."
            )
        }

    memory_block = build_memory_block(relevant_memory)

    combined_context = (
        memory_block + "\n\n" + context
        if memory_block else context
    )

    chain = generation_prompt | llm
    result = chain.invoke({
        "query": state["query"],
        "context": combined_context
    })

    response_text = result.content if hasattr(result, "content") else str(result)

    return {
        **state,
        "context": context,
        "citations": citations,
        "final_response": response_text
    }


# ---------------- OUTPUT FORMATTER ----------------

def format_response_with_metadata(state: GraphState) -> Dict:
    """Format the final response with metadata for display."""
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