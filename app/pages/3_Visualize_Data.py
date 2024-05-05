import os, sys, json, time
from datetime import datetime
import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
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
                                      'llm':True,
                                      'model_options':True,
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

st.info('Visualization is only functional with ChromaDB index type.')

# TODO there's not really a reason for this tab to exist. Consider moving into 2_Chatbot.py app
llm=admin.set_llm(sb,secrets)    # Set the LLM
query_model = admin.get_query_model(sb, secrets)    # Set query model

# Get the viewer
# cert_file_path=os.path.join(paths['base_folder_path'],'app','tls_certificate')
viewer = data_processing.get_or_create_spotlight_viewer()

# Add options
export_file=st.checkbox('Export file?',value=False,help='Export the data, including embeddings to a parquet file')
if export_file:
    file_name=st.text_input('Enter the file name',value=f"{os.path.join(paths['data_folder_path'],sb['index_selected']+'.parquet')}")
cluster_data=st.checkbox('Cluster data?',value=False,help='Cluster the data using the embeddings using KMeans clustering.')
if cluster_data:
    st.markdown('LLM to be used for clustering is set in sidebar.')
    n_clusters=st.number_input('Enter the number of clusters',value=10)
    docs_per_cluster=st.number_input('Enter the number of documents per cluster to generate label',value=10)

if st.button('Visualize'):
    df = data_processing.get_docs_questions_df(
        paths['db_folder_path'],
        sb['index_selected'],
        paths['db_folder_path'],
        sb['index_selected']+'-queries',
        query_model
    )
    if cluster_data:
        df=data_processing.add_clusters(df,n_clusters,
                                        label_llm=llm,
                                        doc_per_cluster=docs_per_cluster)
    if export_file:
        df.to_parquet(file_name)
    st.markdown('Spotlight running on: http://0.0.0.0:9000')
    viewer.show(df, wait=False)