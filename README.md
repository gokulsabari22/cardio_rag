# CARDIO RAG AGENT USING LANGGRAPH
This is a demo for a RAG system that answers questions about heart related issues, symptoms and doubts powered by LangGraph and LangChain.

# Data
Books relted to cardiology has been used 
1. David Adlam, John R. Hampton DM MA DPhil FRCP FFPM FESC, Jo - The ECG Made Easy (2003, Churchill Livingstone) - libgen.li
2. Douglas L. Mann, Douglas P. Zipes, Peter Libby, Robert O. Bonow - Braunwald’s Heart Disease_ A Textbook of Cardiovascular Medicine 1+2 (2014, Saunders) - libgen.li

The books can be found inside the data folder

# Components
The demo uses the following components:

- LLM: Mistral AI (mixtral-8x7b-32768) via LangChain's ChatGroq. Specifically, the LLM is used for three different tasks:
  1. Answer generation
  2. Grading answer hallucinations
  3. Grading answer relevance

- Embeddings: Google Embeddings (text-embedding-004) via LangChain's GoogleGenerativeAIEmbeddings

- Vectorstore: Pinecone Vector DB (via LangChain's Pinecone)

- Web search: Tavily Search (via LangChain's TavilySearchResults)

# Architecture
This project implements a custom RAG architecture that combines ideas from Self RAG and Corrective RAG.

Papers referred:
Self RAG: https://arxiv.org/pdf/2310.11511
Corrective RAG: https://arxiv.org/pdf/2401.15884

The system follows these steps to process and respond to user queries:

1. Query the vector store to retrieve documents relevant to the user's question.

2. Generate an initial response based on the retrieved documents.

3. Evaluate the generated response for factual accuracy:
   - If the response is factually grounded in the documents, proceed to step 4.
   - If the response contains hallucinations, regenerate it (return to step 2).
     This regeneration process is repeated up to N times (configurable by the user).

4. Assess the response for relevance to the user's original question:
   - If relevant, present the response to the user.
   - If not relevant, reformulate the query and restart from step 1.
     This reformulation process is repeated up to N times (configurable by the user).

5. (Optional) If the response remains inaccurate or irrelevant after N attempts:
   - Forward the original user question to a web search engine.
   - Generate a new response based on the web search results.
   - Provide this web-search-based response to the user.

Flowchart for Visual presentation
![flowcahart](https://github.com/user-attachments/assets/e82a86c9-d0b2-41b4-aa4b-45626e28a768)

# Interacting with the agent

Before running the agent, ensure that your environment variables are set in the `.env` file:

```python
PINECONE_API_KEY = <YOUR API KEY>
GOOGLE_API_KEY = <YOUR API KEY>
LANGCHAIN_API_KEY = <YOUR API KEY>
GROQ_API_KEY = <YOUR API KEY>
TAVILY_API_KEY = <YOUR API KEY>