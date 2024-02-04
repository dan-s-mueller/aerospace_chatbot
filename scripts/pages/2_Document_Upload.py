import data_import, setup

import os
import time
import logging
import glob

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from ragatouille import RAGPretrainedModel

import streamlit as st

# Set up the page, enable logging, read environment variables
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_2_document_upload.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set the page title
st.set_page_config(
    page_title='Upload PDFs',
    layout='wide'
)
st.title('Upload PDFs')
# TODO: add database status icons
sb=setup.load_sidebar(config_file='../config/config.json',
                      index_data_file='../config/index_data.json',
                      vector_databases=True,
                      embeddings=True,
                      index_name=True,
                      secret_keys=True)
try:
    secrets=setup.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar
except setup.SecretKeyException as e:
    st.warning(f"{e}")
    st.stop()

# Populate the main screen
logging.info(f'index_type test, {sb["index_type"]}')

if sb["index_type"]=='RAGatouille':
    logging.info('Set hugging face model for queries.')
    query_model=sb['query_model']
elif sb['query_model']=='Openai' or 'Voyage':
    logging.info('Set embeddings model for queries.')
    if sb['query_model']=='Openai':
        query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['query_model']=='Voyage':
        query_model=VoyageEmbeddings(voyage_api_key=secrets['VOYAGE_API_KEY'])
logging.info('Query model set: '+str(query_model))

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
    json_file=st.text_input('Jsonl file',data_folder+'ams_data.jsonl')
    clear_database = st.checkbox('Clear existing database?')
    chunk_method= st.selectbox('Chunk method', ['tiktoken_recursive'], index=0)
    if sb['query_model']=='Openai' or 'ChromaDB':
        # OpenAI will time out if the batch size is too large
        batch_size=st.number_input('Batch size for upsert', min_value=1, step=1, value=100)
    else:
        batch_size=None
    if chunk_method=='tiktoken_recursive':
        chunk_size=st.number_input('Chunk size (tokens)', min_value=1, step=1, value=500)
        chunk_overlap=st.number_input('Chunk overlap (tokens)', min_value=0, step=1, value=0)
    else:
        raise NotImplementedError

# Add a button to run the function
if st.button('Chunk docs to jsonl file'):
    start_time = time.time()  # Start the timer
    data_import.chunk_docs(docs,
                           file=json_file,
                           chunk_method=chunk_method,
                           chunk_size=chunk_size,
                           chunk_overlap=chunk_overlap,
                           use_json=False)
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
if st.button('Load docs into vector database'):
    start_time = time.time()  # Start the timer
    data_import.load_docs(sb['index_type'],
                          docs,
                          query_model=query_model,
                          index_name=sb['index_name'],
                          chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap,
                          use_json=use_json,
                          clear=clear_database,
                          file=json_file,
                          batch_size=batch_size,
                          local_db_path=sb['keys']['LOCAL_DB_PATH'])
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
# Add a button to delete the index
if st.button('Delete existing index'):
    start_time = time.time()  # Start the timer
    data_import.delete_index(sb['index_type'],
                             sb['index_name'],
                             local_db_path=sb['keys']['LOCAL_DB_PATH'])
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")