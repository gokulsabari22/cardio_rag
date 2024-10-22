from document_loader import vectorstore
from graph.state import GraphState
from typing import List
from langchain.schema import Document


def retrieve(state: GraphState) -> List[Document]:

    """
    RETRIEVE DOCUMENTS

    Args:
    state (dict): The current graph state

    Returns:
    state (dict): New key added to state, documents, that contains retrieved documents
    
    """
    print("----------------RETRIEVE-------------------")

    question = state["question"]
    retrieve = vectorstore.similarity_search(query=question, k=1)

    return {"question": question, "documents": retrieve, "web_fallback": True}

if __name__ == "__main__":
    res = retrieve(state={"question": "What are the different values present in human heart"})
    print(res)