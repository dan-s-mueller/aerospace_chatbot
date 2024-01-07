import queries

import os
import time
import logging
import json

import pinecone
import openai

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_community.llms import OpenAI
from langchain_community.llms import HuggingFaceHub

import streamlit as st

# Set up the page, enable logging
from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
with open('config.json', 'r') as f:
    config = json.load(f)
with open('index_data.json', 'r') as f:
    index_data = json.load(f)
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

# Add a sidebar for input options
st.title('Input')

# Vector databases
st.sidebar.title('Vector database')
index_type=st.sidebar.selectbox('Index type', config['databases'], index=0)
logging.info('Index type: '+index_type)

# Embeddings
st.sidebar.title('Embeddings')
if index_type=='RAGatouille':    # Default to selecting hugging face model for RAGatouille, otherwise select alternates
    embedding_type=st.sidebar.selectbox('Embedding type', config['hf_rag_models'], index=0)
else:
    embedding_type=st.sidebar.selectbox('Embedding type', config['embedding_models'], index=0)

if embedding_type=='Openai':
    embedding_name='text-embedding-ada-002'
logging.info('Embedding type: '+embedding_type)
if 'embedding_name' in locals() or 'embedding_name' in globals():
    logging.info('Embedding name: '+embedding_name)

# RAG Type
st.sidebar.title('RAG Type')
rag_type=st.sidebar.selectbox('RAG type', config['rag_types'], index=0)
smart_agent=st.sidebar.checkbox('Smart agent?')
logging.info('RAG type: '+rag_type)
logging.info('Smart agent: '+str(smart_agent))

# Index Name   
index_name=index_data[index_type][embedding_type]
index_name_md=st.sidebar.markdown('Index name: '+index_name)
logging.info('Index name: '+index_name)

# Add input fields in the sidebar
st.sidebar.title('RAG Options')
output_level = st.sidebar.selectbox('Level of Output', ['Concise', 'Detailed'], index=1)
k = st.sidebar.number_input('Number of items per prompt', min_value=1, step=1, value=4)
search_type = st.sidebar.selectbox('Search Type', ['similarity', 'mmr'], index=1)
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
verbose = st.sidebar.checkbox('Verbose output')
chain_type = st.sidebar.selectbox('Chain Type', ['stuff', 'map_reduce'], index=0)
rag_options={'output_level':output_level,
             'k':k,
             'search_type':search_type,
             'temperature':temperature,
             'verbose':verbose,
             'chain_type':chain_type}
logging.info('RAG options: '+str(rag_options))

# LLM
st.sidebar.title('LLM')
llm_model=st.sidebar.selectbox('LLM model', config['llms'], index=0)
if llm_model=='Hugging Face':
    hf_model=st.sidebar.selectbox('Hugging Face model', config['hf_models'], index=0)
logging.info('LLM model: '+llm_model)

# Add a section for secret keys
st.sidebar.title('Secret keys')
if llm_model=='OpenAI' or embedding_type=='Openai':
    OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
    openai.api_key = OPENAI_API_KEY

if llm_model=='Hugging Face':
    HUGGINGFACEHUB_API_TOKEN = st.sidebar.text_input('Hugging Face API Key', type='password')

if embedding_type=='Voyage':
    VOYAGE_API_KEY = st.sidebar.text_input('Voyage API Key', type='password')

if index_type=='Pinecone':
    PINECONE_ENVIRONMENT=st.sidebar.text_input('Pinecone Environment')
    PINECONE_API_KEY=st.sidebar.text_input('Pinecone API Key',type='password')
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')

populate=True

if populate:
    if embedding_type=='Openai':
        embeddings_model=OpenAIEmbeddings(model=embedding_name,openai_api_key=OPENAI_API_KEY)
    elif embedding_type=='Voyage':
        embeddings_model=VoyageEmbeddings(voyage_api_key=VOYAGE_API_KEY)
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
    if output_level == 'Concise':
        out_token = 50
    else:
        out_token = 516
    logging.info('Output tokens: '+str(out_token))

    # Define LLM parameters and qa model object
    if llm_model=='OpenAI':
        llm = OpenAI(temperature=temperature,
                        openai_api_key=OPENAI_API_KEY,
                        max_tokens=out_token)
    elif llm_model=='Hugging Face':
        llm = HuggingFaceHub(repo_id=hf_model,
                            model_kwargs={"temperature": temperature, "max_length": out_token})
    logging.info('LLM model set: '+str(llm))

    qa_model_obj=queries.QA_Model(index_type,
                                  index_name,
                                  embeddings_model,
                                  llm,
                                  k,
                                  search_type,
                                  verbose,
                                  filter_arg=False)

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
                if output_level == 'Concise':
                    out_token = 50
                else:
                    out_token = 516

                # Define LLM parameters and qa model object
                llm = OpenAI(temperature=temperature,
                                openai_api_key=OPENAI_API_KEY,
                                max_tokens=out_token)

                message_id += 1
                st.write('Message: '+str(message_id))
                
                if message_id>1:
                    qa_model_obj=st.session_state['qa_model_obj']
                    qa_model_obj.update_model(llm,
                                        k=k,
                                        search_type=search_type,
                                        verbose=verbose,
                                        filter_arg=filter_toggle)
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