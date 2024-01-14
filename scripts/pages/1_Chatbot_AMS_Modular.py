import queries, setup

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

from langchain_community.llms import OpenAI
from langchain_community.llms import HuggingFaceHub

import streamlit as st

# Set up the page, enable logging
from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app_1.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

populate=False  # Populates the app. Only does so after parameters are set so it doesn't error out.

# Set the page title
st.set_page_config(
    page_title='Aerospace Chatbot: Modular',
)
st.title('Aerospace Mechanisms Chatbot')
with st.expander('''What's under the hood?'''):
    st.markdown('''
    This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
    ''')
filter_toggle=st.checkbox('Filter response with last received sources?')

sb=setup.load_sidebar(config_file='../config/config.json',
                        index_data_file='../config/index_data.json',
                        vector_databases=True,
                        embeddings=True,
                        rag_type=True,
                        index_name=True,
                        llm=True,
                        model_options=True,
                        secret_keys=True)

secrets=setup.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar

populate=True

if populate:
    if sb['embedding_type']=='Openai':
        embeddings_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['embedding_type']=='Voyage':
        embeddings_model=VoyageEmbeddings(model=sb['embedding_name'],voyage_api_key=secrets['VOYAGE_API_KEY'])
    logging.info('Embedding model set: '+str(embeddings_model))

    # Set up chat history
    qa_model_obj = st.session_state.get('qa_model_obj',[])
    message_id = st.session_state.get('message_id', 0)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Process some items
    if sb['model_options']['output_level'] == 'Concise':
        out_token = 50
    else:
        out_token = 516
    logging.info('Output tokens: '+str(out_token))

    # Define LLM parameters and qa model object
    if sb['llm_source']=='OpenAI':
        llm = OpenAI(model_name=sb['llm_model'],
                     temperature=sb['model_options']['temperature'],
                     openai_api_key=secrets['OPENAI_API_KEY'],
                     max_tokens=out_token)
    elif sb['llm_source']=='Hugging Face':
        llm = HuggingFaceHub(repo_id=sb['llm_model'],
                            model_kwargs={"temperature": sb['model_options']['temperature'], "max_length": out_token})
    logging.info('LLM model set: '+str(llm))

    qa_model_obj=queries.QA_Model(sb['index_type'],
                                  sb['index_name'],
                                  embeddings_model,
                                  llm,
                                  k=sb['model_options']['k'],
                                  search_type=sb['model_options']['search_type'],
                                  filter_arg=False)
    logging.info('QA model object set: '+str(qa_model_obj))

    # Display assistant response in chat message container
    if prompt := st.chat_input('Prompt here'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            message_placeholder = st.empty()

            with st.status('Generating response...') as status:
                t_start=time.time()

                # Process some items
                if sb['model_options']['output_level'] == 'Concise':
                    out_token = 50
                else:
                    out_token = 516
                logging.info('Output tokens: '+str(out_token))

                # Define LLM parameters and qa model object
                llm = OpenAI(model_name=sb['llm_model'],
                             temperature=sb['model_options']['temperature'],
                             openai_api_key=secrets['OPENAI_API_KEY'],
                             max_tokens=out_token)
                logging.info('LLM model set: '+str(llm))

                message_id += 1
                st.write('Message: '+str(message_id))
                
                if message_id>1:
                    qa_model_obj=st.session_state['qa_model_obj']
                    qa_model_obj.update_model(llm,
                                              k=sb['model_options']['k'],
                                              search_type=sb['model_options']['search_type'],
                                              filter_arg=filter_toggle)
                    logging.info('QA model object updated: '+str(qa_model_obj))

                    if filter_toggle:
                        filter_list = list(set(item['source'] for item in qa_model_obj.sources[-1]))
                        filter_items=[]
                        for item in filter_list:
                            filter_item={'source': item}
                            filter_items.append(filter_item)
                        filter={'$or':filter_items}
                
                st.write('Searching vector database, generating prompt...')
                qa_model_obj.query_docs(prompt)
                ai_response=qa_model_obj.result['answer']
                message_placeholder.markdown(ai_response)
                t_delta=time.time() - t_start
                status.update(label='Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', state='complete', expanded=False)
        
        st.session_state['qa_model_obj'] = qa_model_obj
        st.session_state['message_id'] = message_id
        st.session_state.messages.append({'role': 'assistant', 'content': ai_response})

else:
    st.warning('Populate secret keys.')
    st.info('Your API-key is not stored in any form by this app. However, for transparency it is recommended to delete your API key once used.')