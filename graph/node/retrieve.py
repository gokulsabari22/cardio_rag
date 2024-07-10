from document_loader import vectorstore
from graph.state import GraphState
from typing import List
from langchain.schema import Document


def retriever(state: GraphState) -> List[Document]:

    """
    RETRIEVE DOCUMENTS

    Args:
    state (dict): The current graph state

    Returns:
    state (dict): New key added to state, documents, that contains retrieved documents
    
    """
    question = state["question"]
    retrieve = vectorstore.similarity_search(query=question, k=1)

    return {"question": question, "documents": retrieve}

if __name__ == "__main__":
    res = retriever(state={"question": "What are the different values present in human heart"})
    print(res)