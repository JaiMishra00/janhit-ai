"""
Agent 4: Response Generation Agent
Generates actionable, cited responses using retrieved context.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict

from models.state import GraphState
from config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL

# Initialize LLM for response generation
llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="not-needed",
    model=LMSTUDIO_MODEL,
    temperature=0.3  # Slightly higher for more natural responses
)

# Response generation prompt
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert assistant that provides actionable, well-cited responses.

CRITICAL RULES:
1. Use ONLY the provided context to answer, If the context contains statutory text, you MAY extract and summarize a definition
even if it is not stated in a single sentence.
2. Cite sources using [Source X] notation after each claim
3. Make responses actionable - tell users what they CAN DO
4. Explain what legal/policy frameworks ENABLE them to do it
5. Be specific and concrete
6. If context is insufficient, say so clearly
7. Structure response with:
   - Direct answer
   - Actionable steps (if applicable)
   - Legal/policy basis (what enables this)
   - Relevant citations

CITATION FORMAT:
- Use [Source 1], [Source 2], etc. for each referenced chunk
- Place citation immediately after the claim
- At the end, list all sources with their metadata

Example response:
"You can request data deletion under GDPR Article 17 [Source 1]. To exercise this right, 
submit a written request to the data controller within their specified timeframe [Source 2].

What enables you to do this:
- GDPR Article 17 grants the 'right to be forgotten' [Source 1]
- Organizations must respond within 30 days [Source 3]

Sources:
[Source 1] GDPR Article 17 (doc_id: gdpr_regulation.pdf, score: 0.92)
[Source 2] Data Protection Guidelines (doc_id: dp_guidelines.pdf, score: 0.88)
[Source 3] Compliance Manual (doc_id: compliance.pdf, score: 0.85)"

Context will be provided with numbered sources.
"""),
    ("human", """
Query: {query}

Context:
{context}

Generate a comprehensive, actionable response with proper citations.""")
])


def prepare_context(matches: List[Dict]) -> tuple[str, List[Dict]]:
    """
    Prepare context string with numbered sources and extract citation metadata.
    
    Args:
        matches: List of retrieved chunks with scores and payloads
        
    Returns:
        Tuple of (formatted_context_string, citation_list)
    """
    if not matches:
        return "No relevant context found.", []
    
    context_parts = []
    citations = []
    
    for idx, match in enumerate(matches, 1):

        # -------- normalize match --------
        if hasattr(match, "payload"):          # ScoredPoint
            payload = match.payload
            score = match.score
        else:                                   # dict
            payload = match.get("payload", {})
            score = match.get("score", 0.0)

        # -------- normalize payload --------
        payload_items = payload if isinstance(payload, list) else [payload]

        for item in payload_items:

            # item can still be ScoredPoint (yes, Qdrant allows nested)
            if hasattr(item, "payload"):
                text = item.payload.get("text", "")
                meta = item.payload
            else:
                text = item.get("text", "")
                meta = item

            context_parts.append(f"[Source {idx}]\n{text}\n")

            citations.append({
                "source_number": idx,
                "doc_id": meta.get("doc_id", "Unknown"),
                "chunk_id": meta.get("chunk_id", 0),
                "score": score if score is not None else 0.0,
                "text_preview": text[:200]
            })


    
    context_string = "\n".join(context_parts)
    return context_string, citations


def generate_response(state: GraphState) -> GraphState:
    """
    Generate final response using retrieved context.
    
    State inputs:
        - query: Original user question
        - matches: Retrieved chunks with scores and payloads
        
    State outputs:
        - context: Formatted context string
        - citations: List of source metadata
        - final_response: Generated answer with citations
        
    Returns:
        Updated state with generated response
    """
    
    print(type(state))

    matches = state.get("matches", [])
    assert isinstance(matches, list)

    
    # Handle case with no matches
    if not matches:
        return {
            **state,
            "context": "",
            "citations": [],
            "final_response": (
                "I couldn't find relevant information in the knowledge base to answer your question. "
                "Please try rephrasing your query or provide more specific details."
            )
        }
    
    # Prepare context and citations
    context, citations = prepare_context(matches)
    
    # Generate response
    try:
        chain = generation_prompt | llm
        result = chain.invoke({
            "query": state["query"],
            "context": context
        })
        
        # Extract text from response
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        return {
        **state,
        "context": context,
        "citations": citations,
        "final_response": response_text
        }

        
    except Exception as e:
        print(f"Error generating response: {e}")
        return {
            **state,
            "context": context,
            "citations": citations,
            "final_response": (
                f"An error occurred while generating the response: {str(e)}\n\n"
                f"However, here are the relevant sources found:\n{context}"
            )
        }


def format_response_with_metadata(state: GraphState) -> Dict:
    """
    Format final output with all metadata for display.
    
    Args:
        state: Final graph state
        
    Returns:
        Formatted response dictionary
    """
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