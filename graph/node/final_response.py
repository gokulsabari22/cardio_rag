from langchain_core.messages import AIMessage
from graph.state import GraphState

def final_response(state: GraphState):
    print("-----------FINAL RESPONSE-----------------")
    return {"message": AIMessage(content=state["generation"])}