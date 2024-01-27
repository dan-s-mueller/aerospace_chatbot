import data_import, setup

import os
import time
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from ragxplorer import RAGxplorer

import streamlit as st

# Set up the page, enable logging 
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_3_visualize_data.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set the page title
st.set_page_config(
    page_title='Visualize Data',
)
st.title('Visualize Data')

sb=setup.load_sidebar(config_file='../config/config.json',
                      index_data_file='../config/index_data.json',
                      vector_databases=True,
                      embeddings=True,
                      index_name=True,
                      secret_keys=True)
secrets=setup.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar

# Populate the main screen
logging.info(f'index_type test, {sb["index_type"]}')

if sb["index_type"]=='RAGatouille':
    raise Exception('Only index type ChromaDB is supported for this function.')
elif sb["index_type"]=='Pinecone':
    raise Exception('Only index type ChromaDB is supported for this function.')
elif sb['query_model']=='Openai' or 'Voyage':
    logging.info('Set embeddings model for queries.')
    if sb['query_model']=='Openai':
        query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['query_model']=='Voyage':
        query_model=VoyageEmbeddings(voyage_api_key=secrets['VOYAGE_API_KEY'])
logging.info('Query model set: '+str(query_model))

st.info('You must have created a database using Document Upload in ChromaDB for this to work.')

# Add an expandable with description of what's going on.
with st.expander("Under the hood"):
    st.markdown('''
                Uses modified version of https://github.com/gabrielchua/RAGxplorer/tree/main?tab=readme-ov-file to connect to existing database created.
                Query size in database: Take a random sample of this size from the database to visualize.
                ''')

# Add a button to run the function
vector_qty=st.number_input('Query size in database', min_value=1, step=10, value=100)
if st.button('Load database'):
    start_time = time.time()  # Start the timer
    
    client = RAGxplorer(embedding_model=sb['embedding_name'])
    client.load_db(path_to_db='../db/chromadb/',index_name=sb['index_name'],verbose=True)

    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time 
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")