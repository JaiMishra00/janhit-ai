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

# Response generation prompt - LESS STRICT VERSION
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

CITATION RULES:
- Add [Source 1], [Source 2], etc. after claims drawn from the context
- You don't need to cite every sentence - focus on key facts and specific claims
- At the end, you may list sources with their metadata

RESPONSE STYLE:
- Be conversational but professional
- Provide direct answers first, then elaborate
- Include practical steps or implications when relevant
- Explain what legal/policy frameworks enable actions, if applicable

Example response:
"An agriculturist under GST Act refers to an individual or entity engaged in agricultural activities [Source 1]. This typically includes cultivation of crops, rearing of livestock, and related farming operations.

The GST framework provides certain exemptions for agriculturists, particularly for the supply of agricultural produce [Source 2]. This means that if you're selling farm produce directly, you may not need to register for GST in many cases.

What this enables:
- Small farmers can sell produce without GST registration burden
- Agricultural cooperatives receive preferential treatment under the act

Sources:
[Source 1] GST Act Section 2(7) - Definition of agriculturist
[Source 2] GST Exemption Notification - Agricultural supplies"

Now answer the user's question using the provided context.
"""),
    ("human", """
Query: {query}

Context:
{context}

Provide a comprehensive answer based on the context above.""")
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
    
    print(f"[CONTEXT] Preparing context from {len(matches)} matches")
    
    for idx, match in enumerate(matches, 1):
        print(f"[CONTEXT] Match {idx} type: {type(match)}")
        
        # Extract from dict format (from retrieval_agent)
        if isinstance(match, dict):
            payload = match.get("payload", {})
            score = match.get("score", 0.0)
            
            print(f"[CONTEXT] Dict match - payload type: {type(payload)}")
            
            # CRITICAL: Handle case where payload might still be a list
            if isinstance(payload, list):
                print(f"[CONTEXT] WARNING: Payload is list with {len(payload)} items")
                if len(payload) > 0:
                    payload = payload[0]
                    print(f"[CONTEXT] Extracted first item, new type: {type(payload)}")
                else:
                    print(f"[CONTEXT] Empty payload list, skipping")
                    continue
                    
        elif hasattr(match, "payload"):  # ScoredPoint object
            payload = match.payload
            score = match.score
            print(f"[CONTEXT] Object match - payload: {type(payload)}")
        else:
            print(f"[CONTEXT] Unknown match format, skipping")
            continue

        # Extract text from payload (now guaranteed to be dict or object)
        text = ""
        doc_id = "Unknown"
        chunk_id = 0
        
        if isinstance(payload, dict):
            text = payload.get("text", "")
            doc_id = payload.get("doc_id", "Unknown")
            chunk_id = payload.get("chunk_id", 0)
            print(f"[CONTEXT] Dict payload - doc_id: {doc_id}, chunk_id: {chunk_id}, text_len: {len(text)}")
        elif hasattr(payload, "__dict__"):
            # Object with attributes
            text = getattr(payload, "text", "")
            doc_id = getattr(payload, "doc_id", "Unknown")
            chunk_id = getattr(payload, "chunk_id", 0)
            print(f"[CONTEXT] Object payload - doc_id: {doc_id}, chunk_id: {chunk_id}, text_len: {len(text)}")
        else:
            print(f"[CONTEXT] Cannot extract from payload type: {type(payload)}")
            print(f"[CONTEXT] Payload content: {payload}")
            continue

        if text and len(text.strip()) > 0:
            context_parts.append(f"[Source {idx}]\n{text}\n")
            
            citations.append({
                "source_number": idx,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": score if score is not None else 0.0,
                "text_preview": text[:200]
            })
            print(f"[CONTEXT] Added source {idx}: {doc_id}::{chunk_id}")
        else:
            print(f"[CONTEXT] Empty text for match {idx}, skipping")
    
    context_string = "\n".join(context_parts)
    print(f"[CONTEXT] Final context: {len(context_string)} chars, {len(citations)} citations")
    
    if len(citations) == 0:
        print("[CONTEXT] WARNING: No citations generated! Check payload structure.")
    
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
    
    matches = state.get("matches", [])
    
    print(f"[GENERATION] Processing {len(matches)} matches")
    
    # Handle case with no matches
    if not matches:
        return {
            **state,
            "context": "",
            "citations": [],
            "final_response": (
                "I couldn't find relevant information in the knowledge base to answer your question. "
                "This might mean:\n"
                "- The information isn't in the current database\n"
                "- The query needs to be rephrased\n"
                "- More specific details would help narrow the search\n\n"
                "Please try rephrasing your query or provide more context."
            )
        }
    
    # Prepare context and citations
    context, citations = prepare_context(matches)
    
    print(f"[GENERATION] Context length: {len(context)} chars")
    print(f"[GENERATION] Citations prepared: {len(citations)}")
    
    # Generate response
    try:
        chain = generation_prompt | llm
        result = chain.invoke({
            "query": state["query"],
            "context": context
        })
        
        # Extract text from response
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        print(f"[GENERATION] Response generated: {len(response_text)} chars")
        print(f"[GENERATION] Response preview:\n{response_text[:500]}...\n")
        
        updated_state = {
            **state,
            "context": context,
            "citations": citations,
            "final_response": response_text
        }
        
        print(f"[GENERATION] State updated - final_response key present: {'final_response' in updated_state}")
        print(f"[GENERATION] Citations count in state: {len(updated_state.get('citations', []))}")
        
        return updated_state
        
    except Exception as e:
        print(f"[GENERATION] Error generating response: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            **state,
            "context": context,
            "citations": citations,
            "final_response": (
                f"An error occurred while generating the response: {str(e)}\n\n"
                f"However, I found {len(citations)} relevant sources. Here's the context:\n\n{context[:500]}..."
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