from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

tool = TavilySearchResults(max_results=4)

def web_search(state: GraphState):
    print("----------------WEB SEARCH---------------------")
    question = state["question"]
    documents = state["documents"]
    search_result = tool.invoke({"query": question})
    combine_result = "\n".join([result["content"] for result in search_result])
    web_search = Document(page_content=combine_result)

    if documents is not None:
        documents.append(web_search)
    else:
        documents = [web_search]

    return {"documents": documents, "web_fallback": False}

if __name__ == "__main__":
    res = web_search(state={"question": "What causes cardiac arrest?", "documents": None})
    print(res)