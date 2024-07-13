# CARDIO RAG AGENT USINH LANGGRAPH
This is a demo for a RAG system that answers questions about heart related issues, symptoms and doubts powered by LangGraph and LangChain.

# Data
Books relted to cardiology has been used 
1. David Adlam, John R. Hampton DM MA DPhil FRCP FFPM FESC, Jo - The ECG Made Easy (2003, Churchill Livingstone) - libgen.li
2. Douglas L. Mann, Douglas P. Zipes, Peter Libby, Robert O. Bonow - Braunwald’s Heart Disease_ A Textbook of Cardiovascular Medicine 1+2 (2014, Saunders) - libgen.li

The books can be found inside the data folder

# Components
The demo uses the following components:

-LLM: Mistral AI (mixtral-8x7b-32768) via LangChain's ChatGroq. Specifically, the LLM is used for three different tasks:
    -answer generation
    -grading answer hallucinations
    -grading answer relevance

-Embeddings: Google Embeddings (text-embedding-004) via LangChain's GoogleGenerativeAIEmbeddings

-Vectorstore: Pinecone Vector DB (via LangChain's Pinecone)

-Web search: Tavily Search (via LangChain's TavilySearchResults)