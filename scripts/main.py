"""
@author: dsmueller3760
Read documents for aerosapce engineers and do things with them.
"""
import os
import glob
import data_import
import queries
from dotenv import load_dotenv,find_dotenv
from langchain.embeddings import OpenAIEmbeddings

# Load API keys
load_dotenv(find_dotenv())
# print(os.getenv('OPENAI_API_KEY'))
# print(os.getenv('PINECONE_ENVIRONMENT'))
# print(os.getenv('PINECONE_API_KEY'))

# Import and instantiate OpenAI embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Pinecone index name
index_name = "langchain-quickstart"

# Find all docs in data folder and import them
data_folder='../data/'
docs = glob.glob(data_folder+'*.pdf')   # Only get the PDFs in the directory
refresh=False
if refresh:
    data_import.load_docs(index_name,embeddings_model,docs)

# List queries you could use
# query = 'What types of lubricants are to be avoided for mechanisms design?'
# query = 'What are examples of harmonic drive gearboxes for aerospace applications?'
# query = 'What types of deployable decelerators are there'
# query = 'What can you tell me about the Orion Side Hatch Design? Please explain any failures and lessons learned in detail'
query = 'What can you tell me about latch mechanism design failures and lessons learned?'
# query = 'What can you tell me about ball-lock mechanism failures? Refer to specific examples.'

response, refs=queries.query_docs(index_name,embeddings_model,query,
                                  k=8,
                                  verbose=True)
print(response)
print('Sources:')
for ref in refs:
    print(ref.metadata)