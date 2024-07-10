from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone()

# INDEX NAME
INDEX_NAME = "cardio-data"

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# loader = PyPDFDirectoryLoader("data/")

# # Load the documents
# document = loader.load()

# # Chunking the text
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=500,
#     chunk_overlap=0
# )

# docs = text_splitter.split_documents(documents=document)

# # Create index if it does not exists
# if INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768, # Embedding model dimension
#         metric="cosine", # Similarity metric
#         spec=ServerlessSpec(
#             cloud='aws', 
#             region='us-east-1'
#         ) 
#     ) 

# # Uploading files to the index
# doc_search = PineconeVectorStore.from_documents(documents=docs, embedding=embeddings, index_name=INDEX_NAME)

vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

question = "What is bicuspid aortic valve"
retriever = vectorstore.similarity_search(query=question, k=1)

print(retriever)