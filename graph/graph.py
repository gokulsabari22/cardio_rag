from graph.state import GraphState
from graph.chain.const import WEBSEARCH, GENERATE, TRANSFORM_QUERY, FINALRESPONSE, RETRIEVE
from graph.chain.const import MAX_RETRIES
from graph.chain.hallucination_grader import hallucination_grader
from graph.chain.answer_grader import answer_grader
from graph.node import retrieve, generate, final_response, websearch, rewrite_question
from langgraph.graph import START, END, StateGraph
from typing import Literal


def grade_generation_v_documents_and_question(state: GraphState, config) -> Literal["generate", "transform_query", "web_search", "finalize_response"]:

    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    web_fallback = state["web_fallback"]
    message = state["message"]
    retries = state["retries"] if state.get("retries") is not None else -1

    # The model has gone through the web search and return to final response
    if not web_fallback:
        return "final_answer"
    
    print("-----------------CHECK HALLUCINATION-------------------")
    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    
    # check hallucination
    if hallucination_score.binary_score == "no":
        return "generate_answer" if retries < MAX_RETRIES else "search_web"
    
    print("-------------------DECISION: GENERATION IS GROUNDED IN DOCUMENTS-------")
    print("-------------------GRADE GENERATION vs QUESTION------------------------")
    answer_score = answer_grader.invoke({"question": question, "generation": generation})

    # check 
    if answer_score.binary_score == "yes":
        print("-------------------DECISION: GENERATION ADDRESSES QUESTION----------------")
        return "final_answer"
    else:
        print("-----------------------DECISION: GENERATION DOES NOT ADDRESS QUESTION-----------------")
        return "rewrite_question" if retries < MAX_RETRIES else "search_web"
    
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(TRANSFORM_QUERY, rewrite_question)
workflow.add_node(WEBSEARCH, websearch)
workflow.add_node(FINALRESPONSE, final_response)

# Build graph
workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GENERATE)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_v_documents_and_question,
    path_map={
        "final_answer": FINALRESPONSE,
        "search_web": WEBSEARCH,
        "generate_answer": GENERATE,
        "rewrite_question": TRANSFORM_QUERY
    }
)

workflow.add_edge(TRANSFORM_QUERY, RETRIEVE)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(FINALRESPONSE, END)

# Compile
app = workflow.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")  