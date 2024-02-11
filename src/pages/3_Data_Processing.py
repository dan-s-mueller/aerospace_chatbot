import admin, data_processing

import time
import logging
import json
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms import HuggingFaceHub

import streamlit as st

from ragxplorer import RAGxplorer

# Set up the page, enable logging 
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_4_data_processing.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set the page title
st.set_page_config(
    page_title='Clean and Question Data',
    layout='wide'
)
st.title('Clean and Question Data')
sb=admin.load_sidebar(config_file='../config/config.json',
                      index_data_file='../config/index_data.json',
                      llm=True,
                      model_options=True,
                      secret_keys=True)
try:
    secrets=admin.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar
except admin.SecretKeyException as e:
    st.warning(f"{e}")
    st.stop()

# Set up session state variables
if 'client' not in st.session_state:
    st.session_state.client = None

# Populate the main screen
# Add an expandable with description of what's going on.
with st.expander("Under the hood",expanded=True):
    st.warning("Under construction")

chunked_file = st.text_input('Chunked raw text file', f'../data/AMS/ams_data-400-0.jsonl')

with st.expander("Process Chunked Data",expanded=True):
    clean_data = st.checkbox('Clean data?', value=True)
    tag_data = st.checkbox('Tag data?', value=True)
    question_data = st.checkbox('Generate questions from data?', value=True)
    if sb['model_options']['output_level'] == 'Concise':
        out_token = 50
    else:
        out_token = 516

    # Define LLM
    if sb['llm_source']=='OpenAI':
        llm = ChatOpenAI(model_name=sb['llm_model'],
                        temperature=sb['model_options']['temperature'],
                        openai_api_key=secrets['OPENAI_API_KEY'],
                        max_tokens=out_token)
    elif sb['llm_source']=='Hugging Face':
        llm = HuggingFaceHub(repo_id=sb['llm_model'],
                            model_kwargs={"temperature": sb['model_options']['temperature'], "max_length": out_token})
    
    if clean_data or tag_data or question_data:
        param_cleaning=None
    if clean_data:
        n_tags=None
    if question_data:
        n_questions=None

    if st.button('Process chunked data'):
        start_time = time.time()  # Start the timer
        
        data_processing.process_chunk(chunked_file,llm,
                  clean_data=False,tag_data=False,question_data=False)

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time 
        st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")