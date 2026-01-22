from typing import List, TypedDict, Dict
from pydantic import BaseModel
from langgraph.graph import StateGraph

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer


# ==================================================
# OUTPUT SCHEMA (Query Decomposition)
# ==================================================

class DecomposedQuery(BaseModel):
    sentences: List[str]


# ==================================================
# GRAPH STATE
# ==================================================

class GraphState(TypedDict):
    # Inputs
    query: str
    documents: List[Dict]   # [{"doc_id": str, "text": str}]

    # Intermediate
    standalone_questions: List[str]
    chunks: List[Dict]      # [{"doc_id", "chunk_id", "text"}]

    # Outputs
    query_embeddings: List[List[float]]
    chunk_embeddings: List[List[float]]


# ==================================================
# MODELS
# ==================================================

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed",
    model="meta-llama-3.1-8b-instruct",
    temperature=0.0
)

embedding_model = LMStudioBgeM3Dense(
    base_url="http://127.0.0.1:1234/v1",
    model="text-embedding-bge-m3"
)

parser = PydanticOutputParser(pydantic_object=DecomposedQuery)


# ==================================================
# PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a query decomposition agent.

Rules:
- Split the query into distinct semantic topics
- Rewrite each topic as a standalone question
- One topic per question
- No overlap
- Do not add new information
- Output ONLY valid JSON
"""),
    ("human", "Query:\n{query}")
])


# ==================================================
# NODES
# ==================================================

def decompose_query(state: GraphState) -> GraphState:
    chain = prompt | llm | parser
    result = chain.invoke({"query": state["query"]})

    return {
        **state,
        "standalone_questions": result.sentences
    }


def chunk_documents(state: GraphState) -> GraphState:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []

    for doc in state["documents"]:
        doc_id = doc["doc_id"]
        text = doc["text"]

        split_texts = splitter.split_text(text)

        for idx, chunk_text in enumerate(split_texts):
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": chunk_text
            })

    return {
        **state,
        "chunks": chunks
    }


def embed_queries(state: GraphState) -> GraphState:
    embeddings = embedding_model.encode(
        state["standalone_questions"],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return {
        **state,
        "query_embeddings": embeddings.tolist()
    }


def embed_documents(state: GraphState) -> GraphState:
    texts = [c["text"] for c in state["chunks"]]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return {
        **state,
        "chunk_embeddings": embeddings.tolist()
    }


# ==================================================
# LANGGRAPH
# ==================================================

graph = StateGraph(GraphState)

graph.add_node("decompose_query", decompose_query)
graph.add_node("chunk_documents", chunk_documents)
graph.add_node("embed_queries", embed_queries)
graph.add_node("embed_documents", embed_documents)

graph.set_entry_point("decompose_query")
graph.add_edge("decompose_query", "chunk_documents")
graph.add_edge("chunk_documents", "embed_queries")
graph.add_edge("embed_queries", "embed_documents")

graph.set_finish_point("embed_documents")

app = graph.compile()


# ==================================================
# LOCAL TEST
# ==================================================

if __name__ == "__main__":
    result = app.invoke({
        "query": "Explain CRISPR, its ethical concerns, and its use in cancer treatment.",
        "documents": [
            {
                "doc_id": "paper_1",
                "text": (
                    "CRISPR is a powerful gene-editing technology that allows scientists "
                    "to modify DNA with high precision. Ethical concerns include germline "
                    "editing, unintended off-target effects, and unequal access. "
                    "In cancer treatment, CRISPR is being explored for immunotherapy, "
                    "including engineered T-cells."
                )
            }
        ]
    })

    print("\nStandalone Questions:")
    for q in result["standalone_questions"]:
        print("-", q)

    print("\nTotal Chunks:", len(result["chunks"]))
    print("Query Embedding Dim:", len(result["query_embeddings"][0]))
    print("Chunk Embedding Dim:", len(result["chunk_embeddings"][0]))
