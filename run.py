"""
Entry point for running the multi-agent RAG system.
Includes response generation with citations.
"""

from graph import app
from Agents.generation_agent import format_response_with_metadata
from pprint import pprint
import sys



def run_query(query: str, files: list = None):
    """
    Execute the RAG pipeline with a user query.
    
    Args:
        query: User's question
        files: Optional list of file paths to process
        
    Returns:
        Final state with retrieval results
    """
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    
    if files:
        print(f"FILES: {files}")
        print("-" * 80)
    
    # Invoke the graph with dict (LangGraph expects dict, not TypedDict instance)
    initial_state = {
        "query": query,
        "files": files
    }

    result = app.invoke(initial_state)
    
    # Display results
    print("\nPIPELINE RESULTS")
    print("-" * 80)
    print(f"Documents extracted: {len(result.get('documents', []))}")
    print(f"Standalone questions: {len(result.get('standalone_questions', []))}")
    print(f"Document chunks: {len(result.get('chunks', []))}")
    print(f"Query embeddings: {len(result.get('query_embeddings', []))}")
    print(f"Chunk embeddings: {len(result.get('chunk_embeddings', []))}")
    print(f"Retrieved matches: {len(result.get('matches', []))}")
    print(f"Citations generated: {len(result.get('citations', []))}")
    
    # Show decomposed questions
    if result.get("standalone_questions"):
        print("\nDECOMPOSED QUESTIONS")
        print("-" * 80)
        for i, q in enumerate(result["standalone_questions"], 1):
            print(f"{i}. {q}")
    
    # Show filters
    if result.get("filters"):
        print("\nFILTERS")
        print("-" * 80)
        print(result["filters"])
    
    # Show generated response
    if result.get("final_response"):
        print("\n" + "=" * 80)
        print("GENERATED RESPONSE")
        print("=" * 80)
        print(result["final_response"])
        print("=" * 80)
    else:
        print("\nWARNING: No final_response in result!")
        print(f"Available keys: {list(result.keys())}")
    
    # Show citations
    if result.get("citations"):
        print("\nCITATIONS")
        print("-" * 80)
        for citation in result["citations"]:
            print(f"\n[Source {citation['source_number']}]")
            print(f"  Document: {citation['doc_id']}")
            print(f"  Chunk ID: {citation['chunk_id']}")
            score = citation.get("score")
            if score is not None:
                print(f"  Relevance Score: {score:.4f}")
            else:
                print("  Relevance Score: N/A")

            if citation.get('text_preview'):
                print(f"  Preview: {citation['text_preview'][:100]}...")
    
    print("\n" + "=" * 80)
    
    return result


def main():
    """Main execution function."""
    
    # Example 1: Text-only query
    # print("\nEXAMPLE 1: Text-only query (no files)\n")
    # run_query(
    #     query="what is an agriculturist in gst act?",
    #     files=None
    # )
    
    # Example 2: Query with document files
    print("\nEXAMPLE 2: Query with PDF files\n")
    run_query(
        query="What is this document? How to fill it?",
        files=["sample.pdf", "scan.jpg"]  # Replace with actual paths
    )
    
    # Example 3: Domain-specific query with filters
    # print("\nEXAMPLE 3: Domain-specific query\n")
    # run_query(
    #     query="how to lodge a FIR complain?",
    #     files=None
    # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)