from typing import List, TypedDict

class GraphState(TypedDict):
    """
    question: question asked by the user
    generation: generation from the LLM model
    web search: does web search required
    documents: documents extracted from the vector database
    
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]