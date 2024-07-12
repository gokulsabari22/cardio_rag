from graph.graph import app
from dotenv import load_dotenv

load_dotenv()

def run_llm(question: str) -> str:
    answer = app.invoke({"question": question})
    for final_answer in answer["message"]:
        return final_answer.content

if __name__ == "__main__":
    res = run_llm("What are the four different valves present in the human heart?")
    print(res)