from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

prompt: ChatPromptTemplate = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()