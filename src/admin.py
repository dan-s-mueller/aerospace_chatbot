import os
import logging
import json

import openai
from pinecone import Pinecone
import chromadb

import streamlit as st

from prompts import TEST_QUERY_PROMPT

# Set up the page, enable logging 
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)

class SecretKeyException(Exception):
    """Exception raised for secret key related errors.

    Attributes:
        message -- explanation of the error
        id -- unique identifier for the error
    """

    def __init__(self, message, id):
        super().__init__(message)
        self.id = id

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
    Loads the sidebar configuration for the chatbot.

    Args:
        config_file (str): The path to the configuration file.
        index_data_file (str): The path to the index data file.
        vector_databases (bool, optional): Whether to include vector databases in the sidebar. Defaults to False.
        embeddings (bool, optional): Whether to include embeddings in the sidebar. Defaults to False.
        rag_type (bool, optional): Whether to include RAG type in the sidebar. Defaults to False.
        index_name (bool, optional): Whether to include index name in the sidebar. Defaults to False.
        llm (bool, optional): Whether to include LLM in the sidebar. Defaults to False.
        model_options (bool, optional): Whether to include model options in the sidebar. Defaults to False.
        secret_keys (bool, optional): Whether to include secret keys in the sidebar. Defaults to False.

    Returns:
        dict: The sidebar configuration.
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
           sb_out['query_model']=st.sidebar.selectbox('Hugging face rag models', 
                                                      databases[sb_out['index_type']]['hf_rag_models'], 
                                                      index=0,
                                                      help="Models listed are compatible with the selected index type.")
        else:
            sb_out['query_model']=st.sidebar.selectbox('Embedding models', 
                                                       databases[sb_out['index_type']]['embedding_models'], 
                                                       index=0,
                                                       help="Models listed are compatible with the selected index type.")

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
            logging.info('RAG type: '+sb_out['rag_type'])

            # TODO: add other advanced RAG types
            # sb_out['smart_agent']=st.sidebar.checkbox('Smart agent?')
            # logging.info('Smart agent: '+str(sb_out['smart_agent']))
    if index_name:
        # Index Name 
        st.sidebar.title('Index Name')  
        sb_out['index_name']=index_data[sb_out['index_type']][sb_out['query_model']]
        st.sidebar.markdown('Index name: '+sb_out['index_name'],help='config/index_data.json contains index names')
        logging.info('Index name: '+sb_out['index_name'])
    if llm:
        # LLM
        st.sidebar.title('LLM')
        st.sidebar.warning('OpenAI LLMs functional, the rest are under construction.')
        sb_out['llm_source']=st.sidebar.selectbox('LLM model', list(llms.keys()), index=0)
        logging.info('LLM source: '+sb_out['llm_source'])
        if sb_out['llm_source']=='OpenAI':
            sb_out['llm_model']=st.sidebar.selectbox('OpenAI model', llms[sb_out['llm_source']]['models'], index=0)
        if sb_out['llm_source']=='Hugging Face':
            sb_out['hf_models']=llms['Hugging Face']['models']
            sb_out['llm_model']=st.sidebar.selectbox('Hugging Face model', 
                                                     [item['model'] for item in llms['Hugging Face']['models']], 
                                                     index=0)
    if model_options:
        # Add input fields in the sidebar
        st.sidebar.title('LLM Options')
        temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=0.1, step=0.1)
        output_level = st.sidebar.selectbox('Level of Output', ['Concise', 'Detailed'], index=1)

        if 'index_type' in sb_out:
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
                                    'temperature':temperature}
        logging.info('Model options: '+str(sb_out['model_options']))
    if secret_keys:
        # Add a section for secret keys
        st.sidebar.title('Secret keys',help='See Home page under Connection Status for status of keys.')
        st.sidebar.markdown('If .env file is in directory, will use that first.')
        sb_out['keys']={}
        if 'llm_source' in sb_out and sb_out['llm_source'] == 'OpenAI':
            sb_out['keys']['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password',help='OpenAI API Key: https://platform.openai.com/api-keys')
        elif 'query_model' in sb_out and sb_out['query_model'] == 'Openai':
            sb_out['keys']['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password',help='OpenAI API Key: https://platform.openai.com/api-keys')
        if 'llm_source' in sb_out and sb_out['llm_source']=='Hugging Face':
            sb_out['keys']['HUGGINGFACEHUB_API_TOKEN'] = st.sidebar.text_input('Hugging Face API Key', type='password',help='Hugging Face API Key: https://huggingface.co/settings/tokens')
        if 'query_model' in sb_out and sb_out['query_model']=='Voyage':
            sb_out['keys']['VOYAGE_API_KEY'] = st.sidebar.text_input('Voyage API Key', type='password',help='Voyage API Key: https://dash.voyageai.com/api-keys')
        if 'index_type' in sb_out and sb_out['index_type']=='Pinecone':
            sb_out['keys']['PINECONE_API_KEY']=st.sidebar.text_input('Pinecone API Key',type='password',help='Pinecone API Key: https://www.pinecone.io/')
        if os.getenv('LOCAL_DB_PATH') is None:
            sb_out['keys']['LOCAL_DB_PATH'] = st.sidebar.text_input('Local Database Path','/data',help='Path to local database (e.g. chroma)')
            os.environ['LOCAL_DB_PATH'] = sb_out['keys']['LOCAL_DB_PATH']
        else:
            sb_out['keys']['LOCAL_DB_PATH'] = os.getenv('LOCAL_DB_PATH')
            st.sidebar.markdown('Local Database Path: '+sb_out['keys']['LOCAL_DB_PATH'],help='From .env file.')
            
    return sb_out

def set_secrets(sb):
    secrets={}

    secrets['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    logging.info('OpenAI API Key: '+str(secrets['OPENAI_API_KEY']))
    if not secrets['OPENAI_API_KEY'] and 'keys' in sb and 'OPENAI_API_KEY' in sb['keys']:
        logging.info('Setting OpenAI API Key from sidebar...')
        secrets['OPENAI_API_KEY'] = sb['keys']['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']
        logging.info('OpenAI API Key: '+str(os.environ['OPENAI_API_KEY']))
        if os.environ['OPENAI_API_KEY']=='':
            raise SecretKeyException('OpenAI API Key is required.','OPENAI_API_KEY_MISSING')
    openai.api_key = secrets['OPENAI_API_KEY']

    secrets['VOYAGE_API_KEY'] = os.getenv('VOYAGE_API_KEY')
    logging.info('Voyage API Key: '+str(secrets['VOYAGE_API_KEY']))
    if not secrets['VOYAGE_API_KEY'] and 'keys' in sb and 'VOYAGE_API_KEY' in sb['keys']:
        logging.info('Setting Voyage API Key from sidebar...')
        secrets['VOYAGE_API_KEY'] = sb['keys']['VOYAGE_API_KEY']
        os.environ['VOYAGE_API_KEY'] = secrets['VOYAGE_API_KEY']
        logging.info('Voyage API Key: '+str(os.environ['VOYAGE_API_KEY']))
        if os.environ['VOYAGE_API_KEY']=='':
            raise SecretKeyException('Voyage API Key is required.','VOYAGE_API_KEY_MISSING')

    secrets['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    logging.info('Pinecone API Key: '+str(secrets['PINECONE_API_KEY']))
    if not secrets['PINECONE_API_KEY'] and 'keys' in sb and 'PINECONE_API_KEY' in sb['keys']:
        logging.info('Setting Pinecone API Key from sidebar...')
        secrets['PINECONE_API_KEY'] = sb['keys']['PINECONE_API_KEY']
        os.environ['PINECONE_API_KEY'] = secrets['PINECONE_API_KEY']
        logging.info('Pinecone API Key: '+str(os.environ['PINECONE_API_KEY']))
        if os.environ['PINECONE_API_KEY']=='':
            raise SecretKeyException('Pinecone API Key is required.','PINECONE_API_KEY_MISSING')

    secrets['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    logging.info('Hugging Face API Key: '+str(secrets['HUGGINGFACEHUB_API_TOKEN']))
    if not secrets['HUGGINGFACEHUB_API_TOKEN'] and 'keys' in sb and 'HUGGINGFACEHUB_API_TOKEN' in sb['keys']:
        logging.info('Setting Hugging Face API Key from sidebar...')
        secrets['HUGGINGFACEHUB_API_TOKEN'] = sb['keys']['HUGGINGFACEHUB_API_TOKEN']
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = secrets['HUGGINGFACEHUB_API_TOKEN']
        logging.info('Hugging Face API Key: '+str(os.environ['HUGGINGFACEHUB_API_TOKEN']))
        if os.environ['HUGGINGFACEHUB_API_TOKEN']=='':
            raise SecretKeyException('Hugging Face API Key is required.','HUGGINGFACE_API_KEY_MISSING')
    return secrets
def test_key_status():
    key_status = {}
    # OpenAI
    if os.getenv('OPENAI_API_KEY') is None:
        key_status['OpenAI API Key'] = {'status': False}
    else:
        key_status['OpenAI API Key'] = {'status': True}

    # Voyage
    if os.getenv('VOYAGE_API_KEY') is None:
        key_status['Voyage API Key'] = {'status': False}
    else:
        key_status['Voyage API Key'] = {'status': True}

    # Pinecone
    if os.getenv('PINECONE_API_KEY') is None:
        key_status['Pinecone API Key'] = {'status': False}
    else:
        key_status['PINECONE_API_KEY'] = {'status': True}
    
    # Hugging Face
    if os.getenv('HUGGINGFACEHUB_API_TOKEN') is None:
        key_status['Hugging Face API Key'] = {'status': False}
    else:
        key_status['Hugging Face API Key'] = {'status': True}

    # Ragatouille local database
        
    return _format_key_status(key_status)

def show_pinecone_indexes():
    if os.getenv('PINECONE_API_KEY') is None:
        pinecone_status = {'status': False, 'message': 'Pinecone API Key is not set.'}
    else:
        pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        indexes=pc.list_indexes()
        if len(indexes)==0:
            pinecone_status = {'status': False, 'message': 'No indexes found'}
        else:
            pinecone_status = {'status': True, 'message': indexes}
    
    return _format_pinecone_status(pinecone_status)
    # return pinecone_status['message'][0]

def show_chroma_collections():
    if os.getenv('LOCAL_DB_PATH') is None:
        chroma_status = {'status': False, 'message': 'Local database path is not set.'}
    else:
        chromadb.Client
        try:
            persistent_client = chromadb.PersistentClient(path=os.getenv('LOCAL_DB_PATH')+'/chromadb')
        except:
            raise ValueError("Chroma vector database needs to be reset. Clear cache.")
        collections=persistent_client.list_collections()
        if len(collections)==0:
            chroma_status = {'status': False, 'message': 'No collections found'}
        else:   
            chroma_status = {'status': True, 'message': collections}
    return _format_chroma_status(chroma_status)

def test_ragatouille_status():
    path=os.getenv('LOCAL_DB_PATH')+'/.ragatouille/colbert/indexes'
    try:
        indexes = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                indexes.append(item)
    except:
        indexes = []

    return _format_ragatouille_status(indexes)

def _format_key_status(key_status:str):
    formatted_status = ""
    for key, value in key_status.items():
        status = value['status']
        if status:
            formatted_status += f"- {key}: :heavy_check_mark:\n"
        else:
            formatted_status += f"- {key}: :x:\n"
    return formatted_status

def _format_pinecone_status(pinecone_status):
    """
    Formats the Pinecone status into a markdown string.

    Args:
        pinecone_status (dict): The Pinecone status dictionary.

    Returns:
        str: The formatted markdown string.
    """
    index_description=''
    if pinecone_status['status']:
        for index in pinecone_status['message']:
            name = index['name']
            state = index['status']['state']
            status = ":heavy_check_mark:"
            index_description += f"- {name}: {state} ({status})\n"
        markdown_string = f"**Pinecone Indexes**\n{index_description}"
    else:
        message = pinecone_status['message']
        markdown_string = f"**Pinecone Indexes**\n- {message}: :x:"
    return markdown_string

def _format_chroma_status(chroma_status):
    """
    Formats the chroma status dictionary into a markdown string.

    Args:
        chroma_status (dict): The chroma status dictionary containing the status and message.

    Returns:
        str: The formatted markdown string.
    """
    collection_description=''
    if chroma_status['status']:
        for index in chroma_status['message']:
            name = index.name
            status = ":heavy_check_mark:"
            collection_description += f"- {name}: ({status})\n"
        markdown_string = f"**ChromaDB Collections**\n{collection_description}"
    else:
        message = chroma_status['message']
        markdown_string = f"**ChromaDB Collections**\n- {message}: :x:"
    return markdown_string

def _format_ragatouille_status(indexes):
    """
    Formats the status of Ragatouille indexes.

    Args:
        indexes (list): List of Ragatouille indexes.

    Returns:
        str: Formatted status of Ragatouille indexes.
    """
    if len(indexes) == 0:
        return "**Ragatouille Indexes**\n- :x: No Ragatouille indexes initialized."
    else:
        index_description = ""
        for index in indexes:
            index_description += f"- {index}: :heavy_check_mark:\n"
        return f"**Ragatouille Indexes**\n{index_description}"