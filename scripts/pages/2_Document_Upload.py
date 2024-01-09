import data_import

import os
import time
import logging
import json
import glob

import pinecone
import openai

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_community.llms import OpenAI
from langchain_community.llms import HuggingFaceHub

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

# Set up the page, enable logging 
from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
with open('config.json', 'r') as f:
    config = json.load(f)
with open('index_data.json', 'r') as f:
    index_data = json.load(f)

# Set the page title
st.set_page_config(
    page_title='Upload PDFs',
)
st.title('Upload PDFs')

# Set up the sidebar
# Vector databases
st.sidebar.title('Vector database')
index_type=st.sidebar.selectbox('Index type', config['databases'], index=0)

# Embeddings
st.sidebar.title('Embeddings')
if index_type=='RAGatouille':    # Default to selecting hugging face model for RAGatouille, otherwise select alternates
    embedding_type=st.sidebar.selectbox('Embedding type', config['hf_rag_models'], index=0)
else:
    embedding_type=st.sidebar.selectbox('Embedding type', config['embedding_models'], index=0)

if embedding_type=='Openai':
    embedding_name='text-embedding-ada-002'

# RAG Type
st.sidebar.title('RAG Type')
rag_type=st.sidebar.selectbox('RAG type', config['rag_types'], index=0)
smart_agent=st.sidebar.checkbox('Smart agent?')
index_name=index_data[index_type][embedding_type]
index_name_md=st.sidebar.markdown('Index name: '+index_name)

# Add a section for secret keys
st.sidebar.title('Secret keys')
if embedding_type=='Openai':
    OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
    openai.api_key = OPENAI_API_KEY

if embedding_type=='Voyage':
    VOYAGE_API_KEY = st.sidebar.text_input('Voyage API Key', type='password')

if index_type=='Pinecone':
    PINECONE_ENVIRONMENT=st.sidebar.text_input('Pinecone Environment')
    PINECONE_API_KEY=st.sidebar.text_input('Pinecone API Key',type='password')
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

# Populate the main screen
if embedding_type=='Openai':
    embeddings_model=OpenAIEmbeddings(model=embedding_name,openai_api_key=OPENAI_API_KEY)
elif embedding_type=='Voyage':
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

# data_import.load_docs(index_name=index_name,
#                       embeddings_model=embeddings_model,
#                       docs=docs,
#                       PINECONE_API_KEY=os.getenv('PINECONE_API_KEY'),
#                       PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT'),
#                       chunk_size=5000,
#                       chunk_overlap=0)

use_json=st.checkbox('Use existing jsonl, if available?',value=True)
clear_database=st.checkbox('Clear existing database?')

# Add a button to run the function
if st.button('Load docs into vector database'):
    start_time = time.time()  # Start the timer
    data_import.load_docs(index_name=index_name,
                          embeddings_model=embeddings_model,
                          docs=docs,
                          chunk_size=5000,
                          chunk_overlap=0,
                          use_json=use_json,
                          clear=clear_database,
                          file=data_folder+'ams_data.jsonl')
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
# Add a button to delete the index
if st.button('Delete existing index'):
    start_time = time.time()  # Start the timer
    data_import.delete_index(index_type,index_name)
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")