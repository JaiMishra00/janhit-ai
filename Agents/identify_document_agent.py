"""
Agent X: Document Identification Agent
Identifies what the uploaded document is, to anchor retrieval on the collection.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

from models.state import GraphState
from config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL


# ---------- Output Schema ----------

class DocumentProfile(BaseModel):
    doc_type: Optional[str] = Field(None, description="Type/name of the document")
    category: Optional[str] = Field(None, description="Domain/category (tax, legal, education, etc.)")
    jurisdiction: Optional[str] = Field(None, description="Country or authority")
    confidence: float = Field(..., ge=0.0, le=1.0)


parser = PydanticOutputParser(pydantic_object=DocumentProfile)

# ---------- LLM ----------

llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="not-needed",
    model=LMSTUDIO_MODEL,
    temperature=0.0
)

# ---------- Prompt ----------

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a document classification agent.

TASK:
Identify what kind of document this is based ONLY on its content.

RULES:
- Do NOT explain
- Do NOT hallucinate details
- If uncertain, leave fields null
- Output ONLY valid JSON matching the schema

Schema:
{format_instructions}
"""),
    ("human", """
Document text:
{text}
""")
]).partial(format_instructions=parser.get_format_instructions())


# ---------- Node ----------

def identify_document(state: GraphState) -> GraphState:
    documents = state.get("documents", [])
    if not documents:
        return {**state, "document_profile": None}

    # Use first document (or concatenate if you want later)
    text = documents[0]["text"][:6000]  # hard cap for safety

    chain = prompt | llm | parser
    profile = chain.invoke({"text": text})

    print("[DOC-ID] Identified document:", profile.dict())

    new_state = dict(state)
    new_state["document_profile"] = profile.dict()
    return new_state

