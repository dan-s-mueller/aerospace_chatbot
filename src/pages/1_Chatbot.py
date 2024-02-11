import admin, queries

import os
import time
import logging
import json

import pinecone
import openai

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from ragatouille import RAGPretrainedModel

import streamlit as st

# Set up the page, enable logging, read environment variables
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_1_chatbot.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set the page title
st.set_page_config(
    page_title='Aerospace Chatbot: Modular',
    layout='wide'
)
# TODO: add database status icons
st.title('Aerospace Chatbot')

sb=admin.load_sidebar(config_file='../config/config.json',
                      index_data_file='../config/index_data.json',
                      vector_databases=True,
                      embeddings=True,
                      rag_type=True,
                      index_name=True,
                      llm=True,
                      model_options=True,
                      secret_keys=True)
try:
    secrets=admin.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar
except admin.SecretKeyException as e:
    st.warning(f"{e}")
    st.stop()

with st.expander('''What's under the hood?'''):
    st.markdown('''
    This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
    Example questions:
    * What are examples of latch failures which have occurred due to improper fitup?
    * What are examples of lubricants which should be avoided for space mechanism applications?
    ''')

# TODO: implement this filtering
# filter_toggle=st.checkbox('Filter response with last received sources?')

# Set up chat history
if 'qa_model_obj' not in st.session_state:
    st.session_state.qa_model_obj = []
    logging.info('QA model object initialized.')
if 'message_id' not in st.session_state:
    st.session_state.message_id = 0
    logging.info('Message ID initialized.')
if 'messages' not in st.session_state:
    st.session_state.messages = []
    logging.info('Messages initialized.')
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
    logging.info('Chat history loaded.')

# Define chat
if prompt := st.chat_input('Prompt here'):
    # User prompt
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    # Assistant response
    with st.chat_message('assistant'):
        message_placeholder = st.empty()

        with st.status('Generating response...') as status:
            t_start=time.time()

            st.session_state.message_id += 1
            st.write('Starting reponse generation for message: '+str(st.session_state.message_id))
            logging.info('Starting reponse generation for message: '+str(st.session_state.message_id))

             # Process some items
            if sb['model_options']['output_level'] == 'Concise':
                out_token = 50
            else:
                out_token = 516
            logging.info('Output tokens: '+str(out_token))
            
            if st.session_state.message_id==1:
                # Define embeddings
                if sb['query_model']=='Openai':
                    query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
                elif sb['query_model']=='Voyage':
                    query_model=VoyageEmbeddings(model=sb['embedding_name'],voyage_api_key=secrets['VOYAGE_API_KEY'])
                elif sb['index_type']=='RAGatouille':
                    query_model=RAGPretrainedModel.from_index(sb['keys']['LOCAL_DB_PATH']+'/.ragatouille/colbert/indexes/'+sb['index_name'])
                logging.info('Query model set: '+str(query_model))

                # Define LLM
                if sb['llm_source']=='OpenAI':
                    llm = ChatOpenAI(model_name=sb['llm_model'],
                                    temperature=sb['model_options']['temperature'],
                                    openai_api_key=secrets['OPENAI_API_KEY'],
                                    max_tokens=out_token)
                elif sb['llm_source']=='Hugging Face':
                    llm = HuggingFaceHub(repo_id=sb['llm_model'],
                                        model_kwargs={"temperature": sb['model_options']['temperature'], "max_length": out_token})
                logging.info('LLM model set: '+str(llm))

                # Initialize QA model object
                if 'search_type' in sb['model_options']: 
                    search_type=sb['model_options']['search_type']
                else:
                    search_type=None
                st.session_state.qa_model_obj=queries.QA_Model(sb['index_type'],
                                                               sb['index_name'],
                                                               query_model,
                                                               llm,
                                                               k=sb['model_options']['k'],
                                                               search_type=search_type,
                                                               filter_arg=False,
                                                               local_db_path=sb['keys']['LOCAL_DB_PATH'])
                logging.info('QA model object set: '+str(st.session_state.qa_model_obj))
            if st.session_state.message_id>1:
                logging.info('Updating model with sidebar settings...')
                # Update LLM
                if sb['llm_source']=='OpenAI':
                    llm = ChatOpenAI(model_name=sb['llm_model'],
                                    temperature=sb['model_options']['temperature'],
                                    openai_api_key=secrets['OPENAI_API_KEY'],
                                    max_tokens=out_token)
                elif sb['llm_source']=='Hugging Face':
                    llm = HuggingFaceHub(repo_id=sb['llm_model'],
                                        model_kwargs={"temperature": sb['model_options']['temperature'], "max_length": out_token})
                logging.info('LLM model set: '+str(llm))

                st.session_state.qa_model_obj.update_model(llm,
                                                           k=sb['model_options']['k'],
                                                           search_type=sb['model_options']['search_type'],
                                                           filter_arg=False)
                logging.info('QA model object updated: '+str(st.session_state.qa_model_obj))
            
            st.write('Searching vector database, generating prompt...')
            logging.info('Searching vector database, generating prompt...')
            st.session_state.qa_model_obj.query_docs(prompt)
            ai_response=st.session_state.qa_model_obj.result['answer'].content
            message_placeholder.markdown(ai_response)
            t_delta=time.time() - t_start
            status.update(label='Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', state='complete', expanded=False)
            
        st.session_state.messages.append({'role': 'assistant', 'content': ai_response})
        logging.info(f'Messaging complete for {st.session_state.message_id}.')

# Add reset button
if st.button('Restart session'):
    st.session_state.qa_model_obj = []
    st.session_state.message_id = 0
    st.session_state.messages = []