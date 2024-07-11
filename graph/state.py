from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class GraphState(TypedDict):
    """
    question: question asked by the user
    generation: generation from the LLM model
    web search: does web search required
    message: 
    documents: documents extracted from the vector database
    
    """
    question: str
    generation: str
    web_fallback: bool
    message: Annotated[list[BaseMessage], add_messages]
    documents: List[str]