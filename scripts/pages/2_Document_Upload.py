import data_import, setup

import os
import time
import logging
import json
import glob

import pinecone
import openai

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_community.llms import OpenAI
from langchain_community.llms import HuggingFaceHub

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

# Set up the page, enable logging 
from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
# with open('../config/config.json', 'r') as f:
#     config = json.load(f)
#     databases = {db['name']: db for db in config['databases']}
#     llms  = {m['name']: m for m in config['llms']}
# with open('../config/index_data.json', 'r') as f:
#     index_data = json.load(f)

# Set the page title
st.set_page_config(
    page_title='Upload PDFs',
)
st.title('Upload PDFs')

sb=setup.load_sidebar(config_file='../config/config.json',
                      index_data_file='../config/index_data.json',
                      vector_databases=True,
                      embeddings=True,
                      index_name=True,
                      secret_keys=True)

# # Set up the sidebar
# # Vector databases
# st.sidebar.title('Vector database')
# index_type=st.sidebar.selectbox('Index type', list(databases.keys()), index=0)
# logging.info('Index type: '+index_type)

# # Embeddings
# st.sidebar.title('Embeddings')
# if index_type=='RAGatouille':    # Default to selecting hugging face model for RAGatouille, otherwise select alternates
#     embedding_type=st.sidebar.selectbox('Hugging face rag models', databases[index_type]['hf_rag_models'], index=0)
# else:
#     embedding_type=st.sidebar.selectbox('Embedding models', databases[index_type]['embedding_models'], index=0)

# if embedding_type=='Openai':
#     embedding_name='text-embedding-ada-002'
# logging.info('Embedding type: '+embedding_type)
# if 'embedding_name' in locals() or 'embedding_name' in globals():
#     logging.info('Embedding name: '+embedding_name)

# # Index
# st.sidebar.title('Index')
# index_name=index_data[index_type][embedding_type]
# index_name_md=st.sidebar.markdown('Index name: '+index_name)

# # Add a section for secret keys
# st.sidebar.title('Secret keys')
# if embedding_type=='Openai':
#     OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
#     openai.api_key = OPENAI_API_KEY

# if embedding_type=='Voyage':
#     VOYAGE_API_KEY = st.sidebar.text_input('Voyage API Key', type='password')

# if index_type=='Pinecone':
#     PINECONE_ENVIRONMENT=st.sidebar.text_input('Pinecone Environment')
#     PINECONE_API_KEY=st.sidebar.text_input('Pinecone API Key',type='password')
#     pinecone.init(
#         api_key=PINECONE_API_KEY,
#         environment=PINECONE_ENVIRONMENT
#     )

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

# Populate the main screen
if sb['embedding_type']=='Openai':
    embeddings_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=OPENAI_API_KEY)
elif sb['embedding_type']=='Voyage':
    embeddings_model=VoyageEmbeddings(voyage_api_key=VOYAGE_API_KEY)
logging.info('Embedding model set: '+str(embeddings_model))


# Upload
# data_folder='../data/FEA/'
index_name_md=st.markdown('Enter a directory relative to the current directory, or an absolute path.')
data_folder = st.text_input('Enter a directory','../data/AMS/')
if not os.path.isdir(data_folder):
    st.error('The entered directory does not exist')
docs = glob.glob(data_folder+'*.pdf')   # Only get the PDFs in the directory
st.markdown('PDFs found: '+str(docs))
st.markdown('Number of PDFs found: ' + str(len(docs)))
logging.info('Docs: '+str(docs))

# Add an expandable box for options
with st.expander("Options"):
    use_json = st.checkbox('Use existing jsonl, if available?', value=True)
    clear_database = st.checkbox('Clear existing database?')
    chunk_method= st.selectbox('Chunk method', ['tiktoken_recursive'], index=0)
    if chunk_method=='tiktoken_recursive':
        chunk_size=st.number_input('Chunk size (tokens)', min_value=1, step=1, value=5000)
        chunk_overlap=st.number_input('Chunk overlap', min_value=0, step=1, value=0)
    else:
        raise NotImplementedError

# Add a button to run the function
if st.button('Load docs into vector database'):
    start_time = time.time()  # Start the timer
    data_import.load_docs(sb['index_type'],
                          docs,
                          embeddings_model,
                          index_name=sb['index_name'],
                          chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap,
                          use_json=use_json,
                          clear=clear_database,
                          file=data_folder+'ams_data.jsonl')
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
# Add a button to delete the index
if st.button('Delete existing index'):
    start_time = time.time()  # Start the timer
    data_import.delete_index(sb['index_type'],sb['index_name'])
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")