import data_import, setup

import os
import time
import logging
import json
import glob

# import pinecone
# import openai

# from langchain_community.vectorstores import Pinecone
# from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_community.llms import OpenAI
from langchain_community.llms import HuggingFaceHub

import streamlit as st

# Set up the page, enable logging 
from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

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

secrets=setup.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar

# Populate the main screen
if sb['embedding_type']=='Openai':
    embeddings_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
elif sb['embedding_type']=='Voyage':
    embeddings_model=VoyageEmbeddings(voyage_api_key=secrets['VOYAGE_API_KEY'])
logging.info('Embedding model set: '+str(embeddings_model))

# Find docs
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
    use_json = st.checkbox('Use existing jsonl, if available (will ignore chunk method, size, and overlap)?', value=True)
    clear_database = st.checkbox('Clear existing database?')
    chunk_method= st.selectbox('Chunk method', ['tiktoken_recursive'], index=0)
    if sb['embedding_type']=='Openai':
        # OpenAI will time out if the batch size is too large
        batch_size=st.number_input('Batch size for upsert', min_value=1, step=1, value=100)
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
                          file=data_folder+'ams_data.jsonl',
                          batch_size=batch_size)
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