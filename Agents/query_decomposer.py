from typing import List, TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ------------------ Output Schema ------------------

class DecomposedQuery(BaseModel):
    sentences: List[str]


# ------------------ Graph State ------------------

class GraphState(TypedDict):
    query: str
    sentences: List[str]


# ------------------ LLM ------------------

llm = Ollama(
    model="llama3:8b-instruct",
    temperature=0.0
)

parser = PydanticOutputParser(pydantic_object=DecomposedQuery)


# ------------------ Prompt ------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a query decomposition agent.

Rules:
- Split the query into distinct semantic topics
- Rewrite each topic as a standalone sentence
- One topic per sentence
- No overlap
- Do not add new information
- Output ONLY valid JSON
"""),
    ("human", "Query:\n{query}")
])


# ------------------ Node ------------------

def decompose_query(state: GraphState) -> GraphState:
    chain = prompt | llm | parser
    result = chain.invoke({"query": state["query"]})
    return {
        "query": state["query"],
        "sentences": result.sentences
    }


# ------------------ Graph ------------------

graph = StateGraph(GraphState)
graph.add_node("decomposer", decompose_query)
graph.set_entry_point("decomposer")
graph.set_finish_point("decomposer")

app = graph.compile()


# ------------------ Local Test ------------------

if __name__ == "__main__":
    out = app.invoke({
        "query": "Explain CRISPR, its ethical concerns, and its use in cancer treatment."
    })
    print(out["sentences"])
