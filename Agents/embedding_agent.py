"""
Agent 2: Embedding Agent
Decomposes queries, chunks documents, and generates embeddings.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.state import GraphState
from models.schemas import DecomposedQuery
from utils.embeddings import LMStudioBgeM3Dense
from config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS
)

# Initialize models
llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="not-needed",
    model=LMSTUDIO_MODEL,
    temperature=0.0
)

embedding_model = LMStudioBgeM3Dense(
    base_url=LMSTUDIO_BASE_URL,
    model=EMBEDDING_MODEL
)

parser = PydanticOutputParser(pydantic_object=DecomposedQuery)

# Query decomposition prompt
decomposition_prompt = ChatPromptTemplate.from_messages([
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


def decompose_query(state: GraphState) -> GraphState:
    """
    Decompose user query into standalone questions.
    
    State inputs:
        - query: User's original question
        
    State outputs:
        - standalone_questions: List of decomposed questions
        
    Returns:
        Updated state with standalone questions
    """
    chain = decomposition_prompt | llm | parser
    result = chain.invoke({"query": state["query"]})
    
    return {
        **state,
        "standalone_questions": [t.question for t in result.topics]
    }


def chunk_documents(state: GraphState) -> GraphState:
    """
    Split documents into chunks for embedding.
    
    State inputs:
        - documents: List of {"doc_id": str, "text": str}
        
    State outputs:
        - chunks: List of {"doc_id": str, "chunk_id": int, "text": str}
        
    Returns:
        Updated state with document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS
    )
    
    chunks = []
    
    for doc in state.get("documents", []):
        doc_id = doc["doc_id"]
        text = doc["text"]
        
        # Skip empty documents
        if not text.strip():
            continue
        
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
    """
    Generate embeddings for decomposed questions.
    
    State inputs:
        - standalone_questions: List of questions
        
    State outputs:
        - query_embeddings: List of embedding vectors
        
    Returns:
        Updated state with query embeddings
    """
    if not state.get("standalone_questions"):
        return {
            **state,
            "query_embeddings": []
        }
    
    embeddings = embedding_model.embed_documents(
        state["standalone_questions"]
    )
    
    return {
        **state,
        "query_embeddings": embeddings
    }


def embed_documents(state: GraphState) -> GraphState:
    """
    Generate embeddings for document chunks.
    
    State inputs:
        - chunks: List of text chunks
        
    State outputs:
        - chunk_embeddings: List of embedding vectors
        
    Returns:
        Updated state with chunk embeddings
    """
    if not state.get("chunks"):
        return {
            **state,
            "chunk_embeddings": []
        }
    
    texts = [c["text"] for c in state["chunks"]]
    embeddings = embedding_model.embed_documents(texts)
    
    return {
        **state,
        "chunk_embeddings": embeddings
    }