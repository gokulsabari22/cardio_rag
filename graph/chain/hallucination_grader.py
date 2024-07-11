from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

class HallucinationGrader(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description = "Answer is grounded in Facts. 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(HallucinationGrader)

system = """

You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no', where 'yes' means that the answer is grounded in / supported by the set of facts.

IF the generation includes code examples, make sure those examples are FULLY present in the set of facts, otherwise always return score 'no'.

"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

if __name__ == "__main__":
    document = [Document(page_content="But sudden cardiac arrest often occurs with no warning. When the heart stops, the lack of oxygen-rich blood can quickly cause death or permanent brain damage. Call 911 or emergency medical services for these symptoms: Chest pain or discomfort. Feeling of a pounding heartbeat. Rapid or irregular heartbeats.\nCardiac arrest is often caused by abnormal heart rhythms due to heart disease, scarring, medications or other factors. Learn about the different types of arrhythmias, their causes and how to prevent or treat them.\nCardiac arrest happens when your heart stops beating or beats too fast due to abnormal electrical impulses. Learn about the common causes, such as arrhythmias, heart attack and drugs, and how to recognize and treat this life-threatening condition.\nOther causes of cardiac arrest include: Scarring of the heart tissue - It may be the result of a prior heart attack or another cause. A heart that's scarred or enlarged from any cause is prone to develop life-threatening ventricular arrhythmias. The first six months after a heart attack is a high-risk period for sudden cardiac arrest in ...")]
    generation = "The different values present in the human heart can be grouped into four broad categories: cellular factors, cardiac factors, extracardiac factors, and physical factors. These values affect the transmission of the cardiac electrical field throughout the body."

    score = hallucination_grader.invoke({"documents": document, "generation": generation})
    print(score.binary_score)