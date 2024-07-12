from graph.state import GraphState
from graph.chain import generation_chain
from langchain.schema import Document
from typing import Dict, Any


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("-----------------GENERATE-----------------")

    question = state["question"]
    documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1 

    generate_answer = generation_chain.invoke({"question": question, "context": documents})

    return {"retries": retries + 1, "generation": generate_answer}

if __name__ == "__main__":
    document = [Document(metadata={'page': 139.0, 'source': 'data\\Douglas L. Mann, Douglas P. Zipes, Peter Libby, Robert O. Bonow - Braunwald’s Heart Disease_ A Textbook of Cardiovascular Medicine 1+2 (2014, Saunders) - libgen.li.pdf'}, page_content='referred to as the volume conductor, which modifies the cardiac \nelectrical field. The contents of the volume conductor are called trans -\nmission factors  to emphasize their effects on transmission of the \ncardiac electrical field throughout the body.\nThey may be grouped into four broad categories—cellular factors, \ncardiac factors, extracardiac factors, and physical factors. Cellular  \nfactors determine the intensity of current fluxes that result from  \nlocal transmembrane potential gradients. Lower concentrations of the \nsodium ion, for example, reduce the intensity of current flow and \nreduce extracellular potentials.\nCardiac  factors affect the relationship of one cardiac cell to another. \nThe two major factors are (1) the more rapid propagation of activation \nalong the length of a fiber than across its width, resulting in greater \ncurrent flow in that direction, and (2) the presence of connective tissue \nbetween cardiac fibers that disrupts efficient electrical coupling of \nadjacent fibers. Recording electrodes oriented along the long axis of \na cardiac fiber register higher potentials than for electrodes oriented \nperpendicular to the long axis. Waveforms recorded from fibers with \nlittle or no intervening connective tissue are narrow in width and \nsmooth in contour, whereas those recorded from tissues with abnor -\nmal fibrosis are prolonged, with prominent notching.\nExtracardiac  factors encompass all the tissues and structures that \nlie between the activation region and the body surface, including  \nthe ventricular walls, intracardiac blood, lungs, skeletal muscle,  \nsubcutaneous fat, and skin. These tissues alter the cardiac field \nbecause of differences in the electrical resistivity of adjacent tissues \nto produce electrical inhomogeneities within the torso. For example, \nintracardiac blood has much lower resistivity (162 Ω cm) than the \nlungs (2150 Ω cm). Differences in torso inhomogeneities can have \nsignificant effects on ECG potentials, especially when the differences \nare exaggerated as in obese persons.\nOther transmission factors reflect basic laws of physics (i.e., physical  \nfactors). Potential magnitudes change in proportion to the square of')]


    res = generate(state={"question": 'What are the different values present in human heart', "documents": document})
    print(res)