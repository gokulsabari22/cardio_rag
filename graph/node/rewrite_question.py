from graph.chain import rewrite_question_chain
from graph.state import GraphState
from typing import Dict, Any


def rewrite_question(state: GraphState) -> Dict[str, Any]:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("----------------REFINE QUESTION-----------------------")

    question = state["question"]

    refined_question = rewrite_question_chain.invoke({"question": question})

    return {"question": refined_question}

if __name__ == "__main__":
    res = rewrite_question(state={"question": "What causes cardiac attack"})
    print(res)