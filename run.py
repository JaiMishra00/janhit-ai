"""
Entry point for running the multi-agent RAG system.
"""

from graph import app
from pprint import pprint
import sys

sys.stdout.reconfigure(encoding="utf-8")



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
    
    # Invoke the graph
    result = app.invoke({
        "query": query,
        "files": files
    })
    
    # Display results
    print("\nPIPELINE RESULTS")
    print("-" * 80)
    print(f"Documents extracted: {len(result.get('documents', []))}")
    print(f"Standalone questions: {len(result.get('standalone_questions', []))}")
    print(f"Document chunks: {len(result.get('chunks', []))}")
    print(f"Query embeddings: {len(result.get('query_embeddings', []))}")
    print(f"Chunk embeddings: {len(result.get('chunk_embeddings', []))}")
    print(f"Retrieved matches: {len(result.get('matches', []))}")
    
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
        pprint(result["filters"])
    
    # Show top matches
    if result.get("matches"):
        print("\nTOP MATCHES")
        print("-" * 80)
        for i, match in enumerate(result["matches"][:5], 1):
            score = match["score"]
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"\n{i}. Score: {score_str}")
            print(f"   ID: {match['id']}")
            if match.get("payload"):
                print(f"   Metadata: {match['payload']}")
    
    print("\n" + "=" * 80)
    
    return result


def main():
    """Main execution function."""
    
    # Example 1: Text-only query
    print("\nEXAMPLE 1: Text-only query (no files)\n")
    run_query(
        query="Who is an agriculturist according to GST acts",
        files=None
    )
    
    # Example 2: Query with document files
    print("\nEXAMPLE 2: Query with PDF files\n")
    run_query(
        query="Summarize the key legal provisions in the document",
        files=["sample.pdf", "scan.jpg"]  # Replace with actual paths
    )
    
    # Example 3: Domain-specific query with filters
    print("\nEXAMPLE 3: Domain-specific query\n")
    run_query(
        query="Find regulations about data privacy enacted in California after 2020",
        files=None
    )


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