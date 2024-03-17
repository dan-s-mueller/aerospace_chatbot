import admin
import data_processing

import streamlit as st
import os

import logging
from dotenv import load_dotenv,find_dotenv

# Set up the page, enable logging, read environment variables
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_Home.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set up page
st.set_page_config(
    page_title="Aerospace Chatbot",
)
st.title("Aerospace Chatbot Homepage")
st.subheader("Aerospace Mechanisms Symposia (AMS)")
st.markdown("""
This space contains chatbots and tools for exploring data in the aerospace mechanisms symposia, using all available papers published since 2000.
Those papers are located here: add link
""")
st.subheader("Code Details")
st.markdown("Code base: https://github.com/dsmueller3760/aerospace_chatbot/tree/rag_study")
st.markdown(
    '''
    API key links:
    * OpenAI: https://platform.openai.com/api-keys
    * Pinecone: https://www.pinecone.io
    * Hugging Face: https://huggingface.co/settings/tokens
    * Voyage: https://dash.voyageai.com/api-keys
    ''')
with st.expander("Connection Status",expanded=True):
    st.markdown("**API keys** (Indicates status of local variable. It does not guarantee the key itself is correct):")
    
    st.markdown(admin.test_key_status())

    # Pinecone
    st.markdown(admin.show_pinecone_indexes())
    try:
        pinecone_indexes = [obj.name for obj in admin.show_pinecone_indexes(format=False)['message']]
        pinecone_index_name=st.selectbox('Pinecone index',pinecone_indexes)
        if st.button('Delete Pinecone index',help='This is permanent!'):
            if pinecone_index_name.endswith("parent-child"):
                rag_type = "Parent-Child"
            elif pinecone_index_name.endswith("summary"):
                rag_type = "Summary"
            else:
                rag_type="Standard"
            data_processing.delete_index('Pinecone',pinecone_index_name,rag_type,local_db_path='../db')
            st.markdown(f"Index {pinecone_index_name} has been deleted.")
    except:
        pass

    # Chroma DB
    st.markdown(admin.show_chroma_collections())
    try:
        chroma_db_collections = [obj.name for obj in admin.show_chroma_collections(format=False)['message']]
        chroma_db_name=st.selectbox('Chroma database',chroma_db_collections)
        if st.button('Delete Pinecone database',help='This is permanent!'):
            if chroma_db_name.endswith("parent-child"):
                rag_type = "Parent-Child"
            elif "-summary-" in chroma_db_name:
                rag_type = "Summary"
            else:
                rag_type="Standard"
            data_processing.delete_index('ChromaDB',chroma_db_name,rag_type,local_db_path='../db')
            st.markdown(f"Database {chroma_db_name} has been deleted.")
    except:
        pass
    
    # Ragatouille
    st.markdown(admin.show_ragatouille_indexes())
    try:
        ragatouille_indexes = [obj.name for obj in admin.show_ragatouille_indexes(format=False)['message']]
        ragatouille_name=st.selectbox('Chroma database',ragatouille_indexes)
        if st.button('Delete Pinecone database',help='This is permanent!'):
            data_processing.delete_index('Ragatouille',ragatouille_name,"Standard",local_db_path='../db')
            st.markdown(f"Index {ragatouille_name} has been deleted.")
    except:
        pass

st.subheader("Running Locally")
'''
It is recommended to run this streamlit app locally for improved performance. The hosted hugging face version is for proof of concept.
You must have poetry installed locally to manage depdenencies. To run locally, clone the repository and run the following commands.
    
    poetry config virtualenvs.in-project true
    poetry install
    source .venv/bin/activate
    cd ./src
    streamlit run Home.py
'''