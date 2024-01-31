import os
import logging
import json

import openai

import streamlit as st

# Set up the page, enable logging 
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)

def load_sidebar(config_file,
                 index_data_file,
                 vector_databases=False,
                 embeddings=False,
                 rag_type=False,
                 index_name=False,
                 llm=False,
                 model_options=False,
                 secret_keys=False):
    """
    Sets up the sidebar based no toggled options. Returns variables with options.
    """
    sb_out={}
    with open(config_file, 'r') as f:
        config = json.load(f)
        databases = {db['name']: db for db in config['databases']}
        llms  = {m['name']: m for m in config['llms']}
        logging.info('Loaded: '+config_file)
    with open(index_data_file, 'r') as f:
        index_data = json.load(f)
        logging.info('Loaded: '+index_data_file)

    if vector_databases:
        # Vector databases
        st.sidebar.title('Vector database')
        sb_out['index_type']=st.sidebar.selectbox('Index type', list(databases.keys()), index=1)
        logging.info('Index type: '+sb_out['index_type'])

    if embeddings:
        # Embeddings
        st.sidebar.title('Embeddings')
        if sb_out['index_type']=='RAGatouille':    # Default to selecting hugging face model for RAGatouille, otherwise select alternates
           sb_out['query_model']=st.sidebar.selectbox('Hugging face rag models', databases[sb_out['index_type']]['hf_rag_models'], index=0)
        else:
            sb_out['query_model']=st.sidebar.selectbox('Embedding models', databases[sb_out['index_type']]['embedding_models'], index=0)

        if sb_out['query_model']=='Openai':
            sb_out['embedding_name']='text-embedding-ada-002'
        elif sb_out['query_model']=='Voyage':
            sb_out['embedding_name']='voyage-02'
        logging.info('Query type: '+sb_out['query_model'])
        if 'embedding_name' in locals() or 'embedding_name' in globals():
            logging.info('Embedding name: '+sb_out['embedding_name'])
    if rag_type:
        if sb_out['index_type']!='RAGatouille': # RAGatouille doesn't have a rag_type
            # RAG Type
            st.sidebar.title('RAG Type')
            sb_out['rag_type']=st.sidebar.selectbox('RAG type', config['rag_types'], index=0)
            sb_out['smart_agent']=st.sidebar.checkbox('Smart agent?')
            logging.info('RAG type: '+sb_out['rag_type'])
            logging.info('Smart agent: '+str(sb_out['smart_agent']))
    if index_name:
        # Index Name 
        st.sidebar.title('Index Name')  
        sb_out['index_name']=index_data[sb_out['index_type']][sb_out['query_model']]
        st.sidebar.markdown('Index name: '+sb_out['index_name'])
        logging.info('Index name: '+sb_out['index_name'])
    if llm:
        # LLM
        st.sidebar.title('LLM')
        sb_out['llm_source']=st.sidebar.selectbox('LLM model', list(llms.keys()), index=0)
        logging.info('LLM source: '+sb_out['llm_source'])
        if sb_out['llm_source']=='OpenAI':
            sb_out['llm_model']=st.sidebar.selectbox('OpenAI model', llms[sb_out['llm_source']]['models'], index=0)
        if sb_out['llm_source']=='Hugging Face':
            sb_out['llm_model']=st.sidebar.selectbox('Hugging Face model', llms[sb_out['llm_source']]['models'], index=0)
    if model_options:
        # Add input fields in the sidebar
        st.sidebar.title('LLM Options')
        temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        output_level = st.sidebar.selectbox('Level of Output', ['Concise', 'Detailed'], index=1)

        st.sidebar.title('Retrieval Options')
        k = st.sidebar.number_input('Number of items per prompt', min_value=1, step=1, value=4)
        if sb_out['index_type']!='RAGatouille':
            search_type = st.sidebar.selectbox('Search Type', ['similarity', 'mmr'], index=0)
            sb_out['model_options']={'output_level':output_level,
                                    'k':k,
                                    'search_type':search_type,
                                    'temperature':temperature}
        else:
            sb_out['model_options']={'output_level':output_level,
                                    'k':k,
                                    'temperature':temperature}
        logging.info('Model options: '+str(sb_out['model_options']))
    if secret_keys:
        # Add a section for secret keys
        st.sidebar.title('Secret keys')
        st.sidebar.markdown('If .env file is in directory, will use that first.')
        sb_out['keys']={}
        if 'llm_source' in sb_out and sb_out['llm_source'] == 'OpenAI':
            sb_out['keys']['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password')
        elif 'query_model' in sb_out and sb_out['query_model'] == 'Openai':
            sb_out['keys']['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password')
        if 'llm_source' in sb_out and sb_out['llm_source']=='Hugging Face':
            sb_out['keys']['HUGGINGFACEHUB_API_TOKEN'] = st.sidebar.text_input('Hugging Face API Key', type='password')
        if 'query_model' in sb_out and sb_out['query_model']=='Voyage':
            sb_out['keys']['VOYAGE_API_KEY'] = st.sidebar.text_input('Voyage API Key', type='password')
        if 'index_type' in sb_out and sb_out['index_type']=='Pinecone':
            sb_out['keys']['PINECONE_API_KEY']=st.sidebar.text_input('Pinecone API Key',type='password')
    return sb_out

def set_secrets(sb):
    """
    Sets secrets from environment file, or from sidebar if not available.
    """
    secrets={}
    secrets['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    openai.api_key = secrets['OPENAI_API_KEY']
    if not secrets['OPENAI_API_KEY'] and 'keys' in sb and 'OPENAI_API_KEY' in sb['keys']:
        secrets['OPENAI_API_KEY'] = sb['keys']['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']
        if os.environ['OPENAI_API_KEY']=='':
            raise Exception('OpenAI API Key is required.')
        openai.api_key = secrets['OPENAI_API_KEY']

    secrets['VOYAGE_API_KEY'] = os.getenv('VOYAGE_API_KEY')
    if not secrets['VOYAGE_API_KEY'] and 'keys' in sb and 'VOYAGE_API_KEY' in sb['keys']:
        secrets['VOYAGE_API_KEY'] = sb['keys']['VOYAGE_API_KEY']
        os.environ['VOYAGE_API_KEY'] = secrets['VOYAGE_API_KEY']
        if os.environ['VOYAGE_API_KEY']=='':
            raise Exception('Voyage API Key is required.')

    secrets['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    if not secrets['PINECONE_API_KEY'] and 'keys' in sb and 'PINECONE_API_KEY' in sb['keys']:
        secrets['PINECONE_API_KEY'] = sb['keys']['PINECONE_API_KEY']
        os.environ['PINECONE_API_KEY'] = secrets['PINECONE_API_KEY']
        if os.environ['PINECONE_API_KEY']=='':
            raise Exception('Pinecone API Key is required.')

    secrets['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not secrets['HUGGINGFACEHUB_API_TOKEN'] and 'keys' in sb and 'HUGGINGFACEHUB_API_TOKEN' in sb['keys']:
        secrets['HUGGINGFACEHUB_API_TOKEN'] = sb['keys']['HUGGINGFACEHUB_API_TOKEN']
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = secrets['HUGGINGFACEHUB_API_TOKEN']
        if os.environ['HUGGINGFACEHUB_API_TOKEN']=='':
            raise Exception('Hugging Face API Key is required.')
    return secrets