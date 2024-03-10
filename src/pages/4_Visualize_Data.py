import admin, data_processing

import json
import time
import logging
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from ragxplorer import RAGxplorer
import chromadb
import streamlit as st
import pandas as pd

# Set up the page, enable logging, read environment variables
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_3_visualize_data.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set the page title
st.set_page_config(
    page_title='Visualize Data',
    layout='wide'
)
st.title('Visualize Data')
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

# Set up session state variables
if 'rx_client' not in st.session_state:
    st.session_state.rx_client = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None

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

# Set the ragxplorer and chromadb clients
st.session_state.rx_client = RAGxplorer(embedding_model=sb['embedding_name'])
st.session_state.chroma_client = chromadb.PersistentClient(path=sb['keys']['LOCAL_DB_PATH']+'/chromadb/')

# Add an expandable with description of what's going on.
with st.expander("Under the hood",expanded=True):
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
            df_export_path = st.text_input('Export file', f'../data/AMS/ams_data-400-0-{vector_qty}.json')
        else:
            df_export_path=st.text_input('Export file', f'../data/AMS/ams_data-400-0-all.json')
    if st.button('Create visualization data'):
        start_time = time.time()  # Start the timer
        
        umap_params={'n_neighbors': 5,'n_components': 2,'random_state':42}  # TODO: expose umap_params as an option
        logging.info(f'embedding_function: {st.session_state.rx_client._chosen_embedding_model}')
        collection=st.session_state.chroma_client.get_collection(name=sb['index_name'],
                                                                 embedding_function=st.session_state.rx_client._chosen_embedding_model)
        st.session_state.rx_client.load_chroma(collection,
                                               umap_params=umap_params,
                                               initialize_projector=True)
        if limit_size:
            st.session_state.rx_client = data_processing.reduce_vector_query_size(st.session_state.rx_client,
                                                                            st.session_state.chroma_client,
                                                                            vector_qty,
                                                                            verbose=True)
        st.session_state.rx_client.run_projector()
        if export_df:
            data_processing.export_data_viz(st.session_state.rx_client,df_export_path)
            
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time 
        st.markdown(f":heavy_check_mark: Created visualization data in {elapsed_time:.2f} seconds")

with st.expander("Visualize data",expanded=True):
    import_data = st.checkbox('Import visualization data?', value=True)
    if import_data:
        import_file = st.file_uploader("Import file", type="json")
        if import_file is None:
            # Use a default file
            import_file_path=st.text_input('Import file',df_export_path)
        else:
            # Use the uploaded file
            import_file_path=st.text_input('Import file',f'../data/AMS/{import_file.name}')
    else:
        import_file_path=None
    
    query = st.text_input('Query', 'What are examples of lubricants which should be avoided for space mechanism applications?')

    if st.button('Visualize data'):
        start_time = time.time()  # Start the timer
        logging.info('Starting visualization for query: '+query)
        if import_data:
            with open(import_file_path, 'r') as f:
                data = json.load(f)
            index_name=data['visualization_index_name']
            umap_params=data['umap_params']
            viz_data=pd.read_json(data['viz_data'], orient='split')
            logging.info('Loaded data from file: '+import_file_path)
            logging.info(f'embedding_function: {st.session_state.rx_client._chosen_embedding_model}')
            collection=st.session_state.chroma_client.get_collection(name=index_name,
                                                                     embedding_function=st.session_state.rx_client._chosen_embedding_model)
            logging.info('Loaded collection: '+index_name)
            st.session_state.rx_client.load_chroma(collection,
                                                umap_params=umap_params,
                                                initialize_projector=True)
            logging.info('Loaded chroma collection: '+index_name)
            fig = st.session_state.rx_client.visualize_query(query,
                                                             import_projection_data=viz_data)
            logging.info('Visualized query: '+query)
        else:
            logging.info('No data loaded from file.')
            fig = st.session_state.rx_client.visualize_query(query)
            logging.info('Visualized query: '+query)
        st.plotly_chart(fig,use_container_width=True)

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time
        st.markdown(f":heavy_check_mark: Created visualization in {elapsed_time:.2f} seconds")