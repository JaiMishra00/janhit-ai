"""
Main LangGraph orchestration for multi-agent RAG system.

Flow:
1. Route extraction based on file presence
2. Extract documents (if files provided) OR skip
3. Decompose query into standalone questions
4. Chunk documents into smaller pieces
5. Embed standalone questions
6. Embed document chunks
7. Generate metadata filters
8. Retrieve and rank relevant chunks
"""

from langgraph.graph import StateGraph, END

from models.state import GraphState
from Agents.extraction_agent import (
    extract_from_files,
    skip_extraction,
    route_extraction
)
from Agents.embedding_agent import (
    decompose_query,
    chunk_documents,
    embed_queries,
    embed_documents
)
from Agents.retrieval_agent import (
    generate_filters,
    retrieve_and_rank
)


def create_graph():
    """
    Create and compile the multi-agent LangGraph.
    
    Returns:
        Compiled LangGraph application
    """
    graph = StateGraph(GraphState)
    
    # ========== REGISTER NODES ==========
    
    # Extraction nodes
    graph.add_node("route_extraction", lambda state: state)
    graph.add_node("extract_documents", extract_from_files)
    graph.add_node("skip_extraction", skip_extraction)
    
    # Embedding nodes
    graph.add_node("decompose_query", decompose_query)
    graph.add_node("chunk_documents", chunk_documents)
    graph.add_node("embed_queries", embed_queries)
    graph.add_node("embed_documents", embed_documents)
    
    # Retrieval nodes
    graph.add_node("generate_filters", generate_filters)
    graph.add_node("retrieve_and_rank", retrieve_and_rank)
    
    # ========== CONFIGURE ROUTING ==========
    
    # Entry point
    graph.set_entry_point("route_extraction")
    
    # Conditional routing for extraction
    graph.add_conditional_edges(
        "route_extraction",
        route_extraction,
        {
            "extract": "extract_documents",
            "skip": "skip_extraction"
        }
    )
    
    # Both extraction paths converge to decompose_query
    graph.add_edge("extract_documents", "decompose_query")
    graph.add_edge("skip_extraction", "decompose_query")
    
    # ========== LINEAR PIPELINE ==========
    
    # Query decomposition → Document chunking
    graph.add_edge("decompose_query", "chunk_documents")
    
    # Document chunking → Query embedding
    graph.add_edge("chunk_documents", "embed_queries")
    
    # Query embedding → Document embedding
    graph.add_edge("embed_queries", "embed_documents")
    
    # Document embedding → Filter generation
    graph.add_edge("embed_documents", "generate_filters")
    
    # Filter generation → Retrieval
    graph.add_edge("generate_filters", "retrieve_and_rank")
    
    # ========== EXIT POINT ==========
    
    graph.add_edge("retrieve_and_rank", END)
    
    # Compile graph
    return graph.compile()


# Create singleton graph instance
app = create_graph()


if __name__ == "__main__":
    # Example usage
    from pprint import pprint
    
    # Test 1: Text-only query (no files)
    print("=" * 60)
    print("TEST 1: Text-only query")
    print("=" * 60)
    
    result = app.invoke({
        "query": "Explain CRISPR, its ethical concerns, and its use in cancer treatment.",
        "files": None
    })
    
    print(f"Standalone questions: {len(result.get('standalone_questions', []))}")
    print(f"Documents: {len(result.get('documents', []))}")
    print(f"Chunks: {len(result.get('chunks', []))}")
    print(f"Query embeddings: {len(result.get('query_embeddings', []))}")
    print(f"Matches: {len(result.get('matches', []))}")
    
    if result.get("standalone_questions"):
        print("\nDecomposed questions:")
        for q in result["standalone_questions"]:
            print(f"  - {q}")
    
    # Test 2: Query with files
    print("\n" + "=" * 60)
    print("TEST 2: Query with files")
    print("=" * 60)
    
    result = app.invoke({
        "query": "Summarize the legal document",
        "files": ["sample.pdf"]  # Replace with actual file path
    })
    
    print(f"Documents extracted: {len(result.get('documents', []))}")
    print(f"Total chunks: {len(result.get('chunks', []))}")
    print(f"Query embeddings: {len(result.get('query_embeddings', []))}")
    print(f"Chunk embeddings: {len(result.get('chunk_embeddings', []))}")
    print(f"Retrieved matches: {len(result.get('matches', []))}")
    
    if result.get("matches"):
        print("\nTop matches:")
        for i, match in enumerate(result["matches"][:3], 1):
            print(f"  {i}. Score: {match['score']:.4f}")
            print(f"     ID: {match['id']}")