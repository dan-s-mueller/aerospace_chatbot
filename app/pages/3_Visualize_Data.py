import os, sys, json, time
from datetime import datetime
import streamlit as st
import pandas as pd
import webbrowser

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
if 'viewer' not in st.session_state:
    st.session_state.viewer = False

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

# Set query model and llm
llm=admin.set_llm(sb,secrets)    # Set the LLM
query_model = admin.get_query_model(sb, secrets)    # Set query model

# Get the viewer
# viewer = data_processing.get_or_create_spotlight_viewer()

# Add options
hf_org_name=st.text_input('Enter the Hugging Face organization name',value='ai-aerospace',help='The organization name on Hugging Face.')
dataset_name=st.text_input('Enter the dataset name',value=sb['index_selected'],help='The name of the dataset to be created on Hugging Face. Will be appended with ac-.')
dataset_name=hf_org_name+'/'+'ac-'+dataset_name
st.markdown(f"The dataset will be created at: {'https://huggingface.co/datasets/'+dataset_name}")
cluster_data=st.checkbox('Cluster data?',value=False,help='Cluster the data using the embeddings using KMeans clustering.')
if cluster_data:
    st.markdown('LLM to be used for clustering is set in sidebar.')
    n_clusters=st.number_input('Enter the number of clusters',value=10)
    docs_per_cluster=st.number_input('Enter the number of documents per cluster to generate label',value=10)

if st.button('Upload dataset to Hugging Face'):
    df = data_processing.get_docs_questions_df(
        paths['db_folder_path'],
        sb['index_selected'],
        paths['db_folder_path'],
        sb['index_selected']+'-queries',
        query_model
    )
    progress_bar = st.progress(0,text='Clustering data...')
    if cluster_data:
        df=data_processing.add_clusters(df,n_clusters,
                                        label_llm=llm,
                                        doc_per_cluster=docs_per_cluster)
    progress_bar.progress(50,text='Exporting to Hugging Face dataset...')
    data_processing.export_to_hf_dataset(df,dataset_name)
    progress_bar.progress(100,text='Export complete!')
    st.session_state.viewer=True

if st.session_state.viewer:
    if st.button('Launch Spotlight data viewer'):
        webbrowser.open_new_tab('https://huggingface.co/spaces/ai-aerospace/aerospace_chatbot_visualize')
        st.markdown('Spotlight viewer launched at: https://huggingface.co/spaces/ai-aerospace/aerospace_chatbot_visualize')