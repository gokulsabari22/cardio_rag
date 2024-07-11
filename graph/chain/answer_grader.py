from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

class AnswerGrader(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(AnswerGrader)

system = """
You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no', where 'yes' means that the answer resolves the question.
"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
])

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

if __name__ == "__main__":
    generated = 'The different values present in the human heart can be grouped into four broad categories: cellular factors, cardiac factors, extracardiac factors, and physical factors. These values affect the transmission of the cardiac electrical field throughout the body.'
    res = answer_grader.invoke({"question": "What causes cardiac arrest?", "generation": generated})
    print(res)