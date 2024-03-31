import os, sys, json, time, logging
from datetime import datetime
from dotenv import load_dotenv,find_dotenv
import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from ragxplorer import RAGxplorer
import chromadb

sys.path.append('../../src/aerospace_chatbot')  # Add package to path
import admin, data_processing

# Page setup
paths,sb,secrets=admin.st_setup_page('Visualize Data',
                                     {'embeddings':True,
                                      'rag_type':True,
                                      'index_name':True,
                                      'secret_keys':True})

# Set up session state variables
if 'rx_client' not in st.session_state:
    st.session_state.rx_client = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None

# Populate the main screen
logging.info(f'index_type test, {sb["index_type"]}')

# Set the query model
if sb["index_type"]=='RAGatouille':
    raise Exception('Only index type ChromaDB is supported for this function.')
elif sb["index_type"]=='Pinecone':
    raise Exception('Only index type ChromaDB is supported for this function.')
elif sb['query_model']=='Openai' or 'Voyage':
    logging.info('Set embeddings model for queries.')
    if sb['query_model']=='Openai':
        query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['query_model']=='Voyage':
        query_model=VoyageAIEmbeddings(model='voyage-2', voyage_api_key=secrets['VOYAGE_API_KEY'])
logging.info('Query model set: '+str(query_model))

st.info('You must have created a database using Document Upload in ChromaDB for this to work.')

# Set the ragxplorer and chromadb clients
st.session_state.rx_client = RAGxplorer(embedding_model=sb['embedding_name'])
st.session_state.chroma_client = chromadb.PersistentClient(path=os.path.join(paths['db_folder_path'],'chromadb'))

# Add an expandable with description of what's going on.
with st.expander("Under the hood",expanded=False):
    st.markdown('''
                Uses modified version of https://github.com/gabrielchua/RAGxplorer/tree/main?tab=readme-ov-file to connect to existing database created.
                Modified version here: https://github.com/dsmueller3760/RAGxplorer/tree/load_db
                Assumes that chroma databases are located in local_db_path variable.
                Query size in database: Take a random sample of this size from the database to visualize.
                ''')

with st.expander("Create visualization data",expanded=True):
    # Add a button to run the function
    limit_size = st.checkbox('Limit size of data visualization?', value=True)
    if limit_size:
        vector_qty=st.number_input('Query size in database', min_value=1, step=10, value=50)
    else:
        vector_qty=None
    export_df = st.checkbox('Export visualization data?', value=True)
    if export_df:
        current_time = datetime.now().strftime("%Y.%m.%d.%H.%M")
        if limit_size:
            df_export_path = st.text_input('Export file', os.path.join(paths['data_folder_path'],'AMS',f'ams_data-400-0-{vector_qty}.json'))
        else:
            df_export_path=st.text_input('Export file', os.path.join(paths['data_folder_path'],'AMS',f'ams_data-400-0-all.json'))

    umap_params={'n_neighbors': 5,'n_components': 2,'random_state':42}

    if st.button('Create visualization data'):
        start_time = time.time()  # Start the timer
        st.session_state.rx_client, st.session_state.chroma_client, temp_index_name = data_processing.create_data_viz(sb['index_selected'],
                            st.session_state.rx_client,
                            st.session_state.chroma_client,
                            umap_params=umap_params,
                            limit_size_qty=vector_qty,
                            df_export_path=df_export_path,
                            show_progress=True)
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time 
        st.markdown(f":heavy_check_mark: Created visualization data in {elapsed_time:.2f} seconds. Visualization database: {temp_index_name}")

with st.expander("Visualize data",expanded=True):
    import_data = st.checkbox('Import visualization data?', value=True)
    if import_data:
        import_file_path=st.text_input('Import file',df_export_path)
    else:
        import_file_path=None
    
    query = st.text_input('Query', 'What are examples of lubricants which should be avoided for space mechanism applications?')

    if st.button('Visualize data'):
        start_time = time.time()  # Start the timer
        st.session_state.rx_client, st.session_state.chroma_client = data_processing.visualize_data(
                            sb['index_selected'],
                            st.session_state.rx_client,
                            st.session_state.chroma_client,
                            query,
                            umap_params=umap_params,
                            import_file=import_file_path)
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time
        st.markdown(f":heavy_check_mark: Created visualization in {elapsed_time:.2f} seconds")