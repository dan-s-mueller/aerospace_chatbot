"""
@author: dsmueller3760
Read documents for aerosapce engineers and do things with them.
"""
import os
import glob
import data_import
import queries
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

# Load API keys
load_dotenv(find_dotenv(),override=True)
# print(os.getenv('PINECONE_ENVIRONMENT'))
# print(os.getenv('PINECONE_API_KEY'))

# OpenAI
llm = OpenAI(temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=os.getenv('OPENAI_API_KEY'))

# Pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT') 
)
index_name = "langchain-quickstart"

# Find all docs in data folder and import them
current_path=os.path.dirname(os.path.abspath(__file__))
data_folder='/../data/'
docs = glob.glob(current_path+data_folder+'*.pdf')   # Only get the PDFs in the directory

refresh=False
if refresh:
    data_import.load_docs(index_name,embeddings_model,docs)

qa=queries.qa_model(index_name,
                    embeddings_model,
                    llm,
                    k=8,
                    search_type='mmr',
                    verbose=False)

query = 'What can you tell me about latch mechanism design failures which have occurred'
# result = qa({"question": query})
queries.query_docs(qa,query)

query_followup='Provide details on volatile spherical joint interfaces'
queries.query_docs(qa,query_followup)