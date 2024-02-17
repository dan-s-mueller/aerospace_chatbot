import admin, data_processing

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
logging.basicConfig(filename='app_2_database_processing.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set the page title
st.set_page_config(
    page_title='Database Processing',
    layout='wide'
)
st.title('Database Processing')
# TODO: add database status icons
sb=admin.load_sidebar(config_file='../config/config.json',
                      index_data_file='../config/index_data.json',
                      vector_databases=True,
                      embeddings=True,
                      rag_type=True,
                      index_name=True,
                      secret_keys=True)
try:
    secrets=admin.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar
except admin.SecretKeyException as e:
    st.warning(f"{e}")
    st.stop()

with st.expander('''What's under the hood?'''):
    st.markdown('''
    This is where you manage databases and process input documents.
    ''')

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
data_folder = st.text_input('Enter a directory','../data/AMS/',help='Enter a directory relative to the current directory, or an absolute path.')
if not os.path.isdir(data_folder):
    st.error('The entered directory does not exist')
docs = glob.glob(data_folder+'*.pdf')   # Only get the PDFs in the directory
st.markdown('PDFs found: '+str(docs))
st.markdown('Number of PDFs found: ' + str(len(docs)))
logging.info('Docs: '+str(docs))
database_appendix=st.text_input('Appendix for database name','ams')

# Add an expandable box for options
with st.expander("Options",expanded=True):
    use_json = st.checkbox('Use existing jsonl?', value=True,help='If checked, the jsonl file will be used for loading into the database.')
    clear_database = st.checkbox('Clear existing database?')
    if sb['query_model']=='Openai' or 'ChromaDB':
        # OpenAI will time out if the batch size is too large
        batch_size=st.number_input('Batch size for upsert', min_value=1, step=1, value=100)
    else:
        batch_size=None
    
    if not use_json:
        chunk_method= st.selectbox('Chunk method', ['tiktoken_recursive'], index=0)
        if chunk_method=='tiktoken_recursive':
            chunk_size=st.number_input('Chunk size (tokens)', min_value=1, step=1, value=500)
            chunk_overlap=st.number_input('Chunk overlap (tokens)', min_value=0, step=1, value=0)
            export_json = st.checkbox('Export jsonl?', value=True,help='If checked, a jsonl file will be generated when you load docs to vector database.')
            if export_json:
                json_file=st.text_input('Jsonl file',data_folder+'ams_data-400-0.jsonl')
        else:
            raise NotImplementedError
    else:
        json_file=st.text_input('Jsonl file',data_folder+'ams_data-400-0.jsonl')
        chunk_method=None
        chunk_size=None
        chunk_overlap=None


# Add a button to run the function
if st.button('Load docs into vector database'):
    start_time = time.time()  # Start the timer
    data_processing.load_docs(sb['index_type'],
                          docs,
                          rag_type=sb['rag_type'],
                          query_model=query_model,
                          index_name=sb['index_name']+'-'+database_appendix,
                          chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap,
                          use_json=use_json,                          
                          file=json_file,
                          clear=clear_database,
                          batch_size=batch_size,
                          local_db_path=sb['keys']['LOCAL_DB_PATH'],
                          show_progress=True)
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.markdown(f":heavy_check_mark: Loaded docs in {elapsed_time:.2f} seconds")

# Add a button to delete the index
if st.button('Delete existing index'):
    start_time = time.time()  # Start the timer
    data_processing.delete_index(sb['index_type'],
                             sb['index_name'],
                             local_db_path=sb['keys']['LOCAL_DB_PATH'])
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.markdown(f":heavy_check_mark: Deleted existing database(s) in {elapsed_time:.2f} seconds")