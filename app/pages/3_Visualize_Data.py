import os, sys, json, time
from datetime import datetime
import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from ragxplorer import RAGxplorer
import chromadb

sys.path.append('../../src/aerospace_chatbot')  # Add package to path
import admin, data_processing

# Page setup
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))
paths,sb,secrets=admin.st_setup_page('Visualize Data',
                                     home_dir,
                                     {'vector_database':True,
                                      'embeddings':True,
                                      'rag_type':True,
                                      'index_selected':True,
                                      'secret_keys':True})

# Set up session state variables
if 'rx_client' not in st.session_state:
    st.session_state.rx_client = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None

# Set the query model
if sb["index_type"]=='RAGatouille':
    raise Exception('Only index type ChromaDB is supported for this function.')
elif sb["index_type"]=='Pinecone':
    raise Exception('Only index type ChromaDB is supported for this function.')
elif sb['query_model']=='OpenAI' or sb['query_model']=='Voyage':
    if sb['query_model']=='OpenAI':
        query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['query_model']=='Voyage':
        query_model=VoyageAIEmbeddings(model='voyage-2', voyage_api_key=secrets['VOYAGE_API_KEY'])

st.info('You must have created a database using Document Upload in ChromaDB for this to work.')

# Set the ragxplorer and chromadb clients
st.session_state.rx_client = RAGxplorer(embedding_model=sb['embedding_name'])
st.session_state.chroma_client = chromadb.PersistentClient(path=os.path.join(paths['db_folder_path'],'chromadb'))

# TODO there's not really a reason for this tab to exist. Move into 2_Chatbot.py app
query_model = admin.get_query_model(sb, secrets)    # Set query model
viewer = data_processing.sl_get_or_create_spotlight_viewer()

if st.button('Visualize'):
    df = data_processing.sl_get_docs_questions_df(
        paths['db_folder_path'],
        sb['index_selected'],
        paths['db_folder_path'],
        sb['index_selected']+'-queries',
        query_model
    )
    viewer.show(df, wait=False)