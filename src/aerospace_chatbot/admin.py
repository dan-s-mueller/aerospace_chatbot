import os
import json
import streamlit as st
from dotenv import load_dotenv,find_dotenv

import openai
from pinecone import Pinecone
import chromadb

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

import nltk # Do before ragatioulle import to avoid logs
nltk.download('punkt', quiet=True)
from ragatouille import RAGPretrainedModel

import fitz
import requests

import data_processing

class SecretKeyException(Exception):
    """Exception raised for secret key related errors.

    Attributes:
        message -- explanation of the error
        id -- unique identifier for the error
    """

    def __init__(self, message, id):
        super().__init__(message)
        self.id = id
class DatabaseException(Exception):
    """Exception raised for database related errors.

    Attributes:
        message -- explanation of the error
        id -- unique identifier for the error
    """

    def __init__(self, message, id):
        super().__init__(message)
        self.id = id

class SidebarManager:
    """Manages the creation, state, and layout of the Streamlit sidebar"""
    
    def __init__(self, config_file):
        self._check_local_db_path()
        self.config = self._load_config(config_file)
        self.sb_out = {}
        self.initialize_session_state()
    
    def _check_local_db_path(self):
        """Check and set local database path if not already set"""
        if not os.environ.get('LOCAL_DB_PATH'):
            local_db_path_input = st.empty()
            warn_db_path = st.warning('Local Database Path is required to initialize. Use an absolute path.')
            local_db_path = local_db_path_input.text_input('Update Local Database Path', help='Path to local database (e.g. chroma).')
            
            if local_db_path:
                os.environ['LOCAL_DB_PATH'] = local_db_path
            else:
                st.stop()
                
            # Clean up UI elements
            local_db_path_input.empty()
            warn_db_path.empty()
    
    def _load_config(self, config_file):
        """Load and parse config file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
            return {
                'databases': {db['name']: db for db in config['databases']},
                'embeddings': {e['name']: e for e in config['embeddings']},
                'llms': {m['name']: m for m in config['llms']},
                'rag_types': config['rag_types']
            }
    
    def initialize_session_state(self):
        """Initialize session state for all sidebar elements"""
        if 'sidebar_state_initialized' not in st.session_state:
            elements = {
                'index': ['index_selected', 'index_type'],
                'embeddings': ['query_model', 'embedding_name', 'embedding_endpoint'],
                'rag': ['rag_type', 'rag_llm_source', 'rag_llm_model', 'rag_endpoint'],
                'llm': ['llm_source', 'llm_model', 'llm_endpoint'],
                'model_options': ['temperature', 'output_level', 'k', 'search_type'],
                'api_keys': ['openai_key', 'hf_key', 'voyage_key', 'pinecone_key']
            }
            
            for group in elements.values():
                for element in group:
                    if f'{element}_disabled' not in st.session_state:
                        st.session_state[f'{element}_disabled'] = False
                    if f'{element}_value' not in st.session_state:
                        st.session_state[f'{element}_value'] = None
            
            st.session_state.sidebar_state_initialized = True
    
    def _render_index_selection(self):
        """Render index selection section"""
        st.sidebar.title('Index Selected')
        available_indexes = _get_available_indexes(
            self.sb_out['index_type'], 
            self.sb_out['embedding_name'], 
            self.sb_out['rag_type']
        )
        if not available_indexes:
            raise DatabaseException('No compatible collections found.', 'NO_COMPATIBLE_COLLECTIONS')
        
        self.sb_out['index_selected'] = st.sidebar.selectbox(
            'Index selected', 
            available_indexes,
            disabled=st.session_state.index_selected_disabled,
            help='Select the index to use for the application.'
        )

    def _render_vector_database(self):
        """Render vector database section"""
        st.sidebar.title('Vector Database Type')
        self.sb_out['index_type'] = st.sidebar.selectbox(
            'Index type',
            list(self.config['databases'].keys()),
            disabled=st.session_state.index_type_disabled,
            help='Select the type of index to use.'
        )

    def _render_embeddings(self):
        """Render embeddings configuration section"""
        st.sidebar.title('Embeddings', 
                        help='See embedding leaderboard here for performance overview: https://huggingface.co/spaces/mteb/leaderboard')
        
        if self.sb_out['index_type'] == 'RAGatouille':
            self.sb_out['query_model'] = st.sidebar.selectbox(
                'Hugging face rag models',
                self.config['databases'][self.sb_out['index_type']]['embedding_models'],
                disabled=st.session_state.query_model_disabled,
                help="Models listed are compatible with the selected index type."
            )
            self.sb_out['embedding_name'] = self.sb_out['query_model']
        else:
            self.sb_out['query_model'] = st.sidebar.selectbox(
                'Embedding model family',
                self.config['databases'][self.sb_out['index_type']]['embedding_models'],
                disabled=st.session_state.query_model_disabled,
                help="Model provider."
            )
            self.sb_out['embedding_name'] = st.sidebar.selectbox(
                'Embedding model',
                self.config['embeddings'][self.sb_out['query_model']]['embedding_models'],
                disabled=st.session_state.embedding_name_disabled,
                help="Models listed are compatible with the selected index type."
            )
            
            if self.sb_out['embedding_name'] == "Dedicated Endpoint":
                self.sb_out['embedding_hf_endpoint'] = st.sidebar.text_input(
                    'Dedicated endpoint URL',
                    '',
                    disabled=st.session_state.embedding_endpoint_disabled,
                    help='See Hugging Face configuration for endpoint.'
                )
            else:
                self.sb_out['embedding_hf_endpoint'] = None

    def _render_rag_type(self):
        """Render RAG type configuration section"""
        st.sidebar.title('RAG Type')
        if self.sb_out['index_type'] == 'RAGatouille':
            self.sb_out['rag_type'] = st.sidebar.selectbox(
                'RAG type',
                ['Standard'],
                disabled=st.session_state.rag_type_disabled,
                help='Only Standard is available for RAGatouille.'
            )
        else:
            self.sb_out['rag_type'] = st.sidebar.selectbox(
                'RAG type',
                self.config['rag_types'],
                disabled=st.session_state.rag_type_disabled,
                help='Parent-Child is for parent-child RAG. Summary is for summarization RAG.'
            )
            
            if self.sb_out['rag_type'] == 'Summary':
                self._render_rag_llm_config()

    def _render_rag_llm_config(self):
        """Render RAG LLM configuration section"""
        self.sb_out['rag_llm_source'] = st.sidebar.selectbox(
            'RAG LLM model',
            list(self.config['llms'].keys()),
            disabled=st.session_state.rag_llm_source_disabled,
            help='Select the LLM model for RAG.'
        )
        
        self._render_llm_model_selection('rag_')

    def _render_llm(self):
        """Render LLM configuration section"""
        st.sidebar.title('LLM', 
                        help='See LLM leaderboard here for performance overview: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard')
        
        self.sb_out['llm_source'] = st.sidebar.selectbox(
            'LLM model',
            list(self.config['llms'].keys()),
            disabled=st.session_state.llm_source_disabled,
            help='Select the LLM model for the application.'
        )
        
        self._render_llm_model_selection()

    def _render_llm_model_selection(self, prefix=''):
        """Render LLM model selection based on source"""
        source = f"{prefix.replace('_', '')}llm_source"
        model = f"{prefix.replace('_', '')}llm_model"
        endpoint = f"{prefix.replace('_', '')}hf_endpoint"
        
        if self.sb_out[source] == 'OpenAI':
            self.sb_out[model] = st.sidebar.selectbox(
                'OpenAI model',
                self.config['llms'][self.sb_out[source]]['models'],
                disabled=st.session_state[f"{model}_disabled"],
                help='Select the OpenAI model for the application.'
            )
        elif self.sb_out[source] == 'Hugging Face':
            self.sb_out[model] = st.sidebar.selectbox(
                'Hugging Face model',
                self.config['llms']['Hugging Face']['models'],
                disabled=st.session_state[f"{model}_disabled"],
                help='Select the Hugging Face model for the application.'
            )
            self.sb_out[endpoint] = 'https://api-inference.huggingface.co/v1'
            
            if self.sb_out[model] == "Dedicated Endpoint":
                endpoint_url = st.sidebar.text_input(
                    'Dedicated endpoint URL',
                    '',
                    disabled=st.session_state[f"{endpoint}_disabled"],
                    help='See Hugging Face configuration for endpoint.'
                )
                self.sb_out[endpoint] = endpoint_url + '/v1/'
        elif self.sb_out[source] == 'LM Studio (local)':
            self.sb_out[model] = st.sidebar.text_input(
                'Local host URL',
                'http://localhost:1234/v1',
                disabled=st.session_state[f"{model}_disabled"],
                help='See LM studio configuration for local host URL.'
            )
            st.sidebar.warning('You must load a model in LM studio first for this to work.')

    def _render_model_options(self):
        """Render model options section"""
        st.sidebar.title('LLM Options')
        temperature = st.sidebar.slider(
            'Temperature',
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.1,
            disabled=st.session_state.temperature_disabled,
            help='Temperature for LLM.'
        )
        
        output_level = st.sidebar.number_input(
            'Max output tokens',
            min_value=50,
            step=10,
            value=1000,
            disabled=st.session_state.output_level_disabled,
            help='Max output tokens for LLM. Concise: 50, Verbose: 1000. Limit depends on model.'
        )
        
        st.sidebar.title('Retrieval Options')
        k = st.sidebar.number_input(
            'Number of items per prompt',
            min_value=1,
            step=1,
            value=4,
            disabled=st.session_state.k_disabled,
            help='Number of items to retrieve per query.'
        )
        
        if self.sb_out['index_type'] != 'RAGatouille':
            search_type = st.sidebar.selectbox(
                'Search Type',
                ['similarity', 'mmr'],
                disabled=st.session_state.search_type_disabled,
                help='Select the search type for the application.'
            )
            self.sb_out['model_options'] = {
                'output_level': output_level,
                'k': k,
                'search_type': search_type,
                'temperature': temperature
            }
        else:
            self.sb_out['model_options'] = {
                'output_level': output_level,
                'k': k,
                'search_type': None,
                'temperature': temperature
            }

    def _render_secret_keys(self):
        """Render secret keys configuration section"""
        self.sb_out['keys'] = {}
        st.sidebar.title('Secret keys', help='See Home page under Connection Status for status of keys.')
        st.sidebar.markdown('If .env file is in directory, will use that first.')
        
        # Only show relevant API key inputs based on selected models
        if ('llm_source' in self.sb_out and self.sb_out['llm_source'] == 'OpenAI') or \
           ('query_model' in self.sb_out and self.sb_out['query_model'] == 'OpenAI'):
            self.sb_out['keys']['OPENAI_API_KEY'] = st.sidebar.text_input(
                'OpenAI API Key',
                type='password',
                disabled=st.session_state.openai_key_disabled,
                help='OpenAI API Key: https://platform.openai.com/api-keys'
            )
        
        if 'llm_source' in self.sb_out and self.sb_out['llm_source'] == 'Anthropic':
            self.sb_out['keys']['ANTHROPIC_API_KEY'] = st.sidebar.text_input(
                'Anthropic API Key',
                type='password',
                disabled=st.session_state.anthropic_key_disabled,
                help='Anthropic API Key: https://console.anthropic.com/settings/keys'
            )
        
        if 'query_model' in self.sb_out and self.sb_out['query_model'] == 'Voyage':
            self.sb_out['keys']['VOYAGE_API_KEY'] = st.sidebar.text_input(
                'Voyage API Key',
                type='password',
                disabled=st.session_state.voyage_key_disabled,
                help='Voyage API Key: https://dash.voyageai.com/api-keys'
            )
        
        if 'index_type' in self.sb_out and self.sb_out['index_type'] == 'Pinecone':
            self.sb_out['keys']['PINECONE_API_KEY'] = st.sidebar.text_input(
                'Pinecone API Key',
                type='password',
                disabled=st.session_state.pinecone_key_disabled,
                help='Pinecone API Key: https://www.pinecone.io/'
            )
        
        if ('llm_source' in self.sb_out and self.sb_out['llm_source'] == 'Hugging Face') or \
           ('query_model' in self.sb_out and self.sb_out['query_model'] == 'Hugging Face'):
            self.sb_out['keys']['HUGGINGFACEHUB_API_TOKEN'] = st.sidebar.text_input(
                'Hugging Face API Key',
                type='password',
                disabled=st.session_state.hf_key_disabled,
                help='Hugging Face API Key: https://huggingface.co/settings/tokens'
            )
        
        # Set secrets and environment variables
        try:
            self.secrets = set_secrets(self.sb_out)
        except SecretKeyException as e:
            st.warning(f"{e}")
            st.stop()

    def render_sidebar(self, vector_database=False, embeddings=False, rag_type=False,
                      index_selected=False, llm=False, model_options=False, secret_keys=False):
        """Render the complete sidebar based on enabled options"""
        try:
            if vector_database:
                self._render_vector_database()
                
                if embeddings:
                    self._render_embeddings()
                
                if rag_type:
                    self._render_rag_type()
                
                if index_selected and embeddings and rag_type:
                    self._render_index_selection()
                
                if llm:
                    self._render_llm()
                
                if model_options:
                    self._render_model_options()
            
            if secret_keys:
                self._render_secret_keys()
        except DatabaseException as e:
            st.error(f"No index available, create a new one with the sidebar parameters you've selected: {e}")
            st.stop()
        return self.sb_out

    def get_paths(self, home_dir):
        """Get application paths"""
        return {
            'base_folder_path': home_dir,
            'db_folder_path': os.path.join(home_dir, os.getenv('LOCAL_DB_PATH')),
            'data_folder_path': os.path.join(home_dir, 'data')
        }

    def get_secrets(self):
        """Load and return secrets from environment"""
        load_dotenv(find_dotenv())
        return {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'HUGGINGFACEHUB_API_TOKEN': os.getenv('HUGGINGFACEHUB_API_TOKEN'),
            'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY'),
            'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY')
        }

def _get_available_indexes(index_type, embedding_name, rag_type):
    """Helper function to get available indexes based on current settings"""
    name = []
    base_name = embedding_name.replace('/', '-').lower()
    
    if index_type == 'ChromaDB':
        indices = show_chroma_collections(format=False)
        if not indices['status']:
            return []
        
        for index in indices['message']:
            if not index.name.startswith(base_name) or index.name.endswith('-queries'):
                continue
                
            if (rag_type == 'Parent-Child' and index.name.endswith('-parent-child')) or \
               (rag_type == 'Summary' and index.name.endswith('-summary')) or \
               (rag_type == 'Standard' and not index.name.endswith(('-parent-child', '-summary'))):
                name.append(index.name)
                
    elif index_type == 'Pinecone':
        indices = show_pinecone_indexes(format=False)
        if not indices['status']:
            return []
            
        for index in indices['message']:
            if not index['name'].startswith(base_name) or index['name'].endswith('-queries'):
                continue
                
            if (rag_type == 'Parent-Child' and index['name'].endswith('-parent-child')) or \
               (rag_type == 'Summary' and index['name'].endswith('-summary')) or \
               (rag_type == 'Standard' and not index['name'].endswith(('-parent-child', '-summary'))):
                name.append(index['name'])
                
    elif index_type == 'RAGatouille':
        indices = show_ragatouille_indexes(format=False)
        if not indices['status']:
            return []
            
        for index in indices['message']:
            if index.startswith(base_name):
                name.append(index)
    
    return name

def set_secrets(sb):
    """
    Sets the secrets for various API keys by retrieving them from the environment variables or the sidebar.

    Args:
        sb (dict): The sidebar data containing the API keys.

    Returns:
        dict: A dictionary containing the set API keys.

    Raises:
        SecretKeyException: If any of the required API keys are missing.
    """
    secrets={}

    secrets['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    if not secrets['OPENAI_API_KEY'] and 'keys' in sb and 'OPENAI_API_KEY' in sb['keys']:
        secrets['OPENAI_API_KEY'] = sb['keys']['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']
        if os.environ['OPENAI_API_KEY']=='':
            raise SecretKeyException('OpenAI API Key is required.','OPENAI_API_KEY_MISSING')
    openai.api_key = secrets['OPENAI_API_KEY']

    secrets['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    if not secrets['ANTHROPIC_API_KEY'] and 'keys' in sb and 'ANTHROPIC_API_KEY' in sb['keys']:
        secrets['ANTHROPIC_API_KEY'] = sb['keys']['ANTHROPIC_API_KEY']
        os.environ['ANTHROPIC_API_KEY'] = secrets['ANTHROPIC_API_KEY']
        if os.environ['ANTHROPIC_API_KEY']=='':
            raise SecretKeyException('Anthropic API Key is required.','ANTHROPIC_API_KEY_MISSING')

    secrets['VOYAGE_API_KEY'] = os.getenv('VOYAGE_API_KEY')
    if not secrets['VOYAGE_API_KEY'] and 'keys' in sb and 'VOYAGE_API_KEY' in sb['keys']:
        secrets['VOYAGE_API_KEY'] = sb['keys']['VOYAGE_API_KEY']
        os.environ['VOYAGE_API_KEY'] = secrets['VOYAGE_API_KEY']
        if os.environ['VOYAGE_API_KEY']=='':
            raise SecretKeyException('Voyage API Key is required.','VOYAGE_API_KEY_MISSING')

    secrets['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    if not secrets['PINECONE_API_KEY'] and 'keys' in sb and 'PINECONE_API_KEY' in sb['keys']:
        secrets['PINECONE_API_KEY'] = sb['keys']['PINECONE_API_KEY']
        os.environ['PINECONE_API_KEY'] = secrets['PINECONE_API_KEY']
        if os.environ['PINECONE_API_KEY']=='':
            raise SecretKeyException('Pinecone API Key is required.','PINECONE_API_KEY_MISSING')

    secrets['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not secrets['HUGGINGFACEHUB_API_TOKEN'] and 'keys' in sb and 'HUGGINGFACEHUB_API_TOKEN' in sb['keys']:
        secrets['HUGGINGFACEHUB_API_TOKEN'] = sb['keys']['HUGGINGFACEHUB_API_TOKEN']
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = secrets['HUGGINGFACEHUB_API_TOKEN']
        if os.environ['HUGGINGFACEHUB_API_TOKEN']=='':
            raise SecretKeyException('Hugging Face API Key is required.','HUGGINGFACE_API_KEY_MISSING')
    return secrets
def test_key_status():
    """
    Check the status of various API keys based on environment variables.

    Returns:
        dict: A dictionary containing the status of each API key.
    """
    key_status = {}
    # OpenAI
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('OPENAI_API_KEY') == '':
        key_status['OpenAI API Key'] = {'status': False}
    else:
        key_status['OpenAI API Key'] = {'status': True}

    # Voyage
    if os.getenv('VOYAGE_API_KEY') is None or os.getenv('VOYAGE_API_KEY') == '':
        key_status['Voyage API Key'] = {'status': False}
    else:
        key_status['Voyage API Key'] = {'status': True}

    # Pinecone
    if os.getenv('PINECONE_API_KEY') is None or os.getenv('PINECONE_API_KEY') == '':
        key_status['Pinecone API Key'] = {'status': False}
    else:
        key_status['PINECONE_API_KEY'] = {'status': True}
    
    # Hugging Face
    if os.getenv('HUGGINGFACEHUB_API_TOKEN') is None or os.getenv('HUGGINGFACEHUB_API_TOKEN') == '':
        key_status['Hugging Face API Key'] = {'status': False}
    else:
        key_status['Hugging Face API Key'] = {'status': True}

    return _format_key_status(key_status)
def set_llm(sb, secrets, type='prompt'):
    """
    Sets up and returns a language model (LLM) based on the provided parameters.

    Args:
        sb (dict): The configuration settings for the chatbot.
        secrets (dict): The secret keys and tokens required for authentication.
        type (str, optional): The type of LLM to set up.

    Returns:
        ChatAnthropic or ChatOpenAI: The configured language model.

    Raises:
        ValueError: If an invalid LLM source is specified.

    """
    if type == 'prompt':  # use for prompting in chat applications
        if sb['llm_source'] == 'OpenAI':
            llm = ChatOpenAI(model_name=sb['llm_model'],
                             temperature=sb['model_options']['temperature'],
                             openai_api_key=secrets['OPENAI_API_KEY'],
                             max_tokens=sb['model_options']['output_level'],
                             tags=[sb['llm_model']])
        elif sb['llm_source'] == 'Anthropic':
            llm = ChatAnthropic(model=sb['llm_model'],
                                temperature=sb['model_options']['temperature'],
                                api_key=secrets['ANTHROPIC_API_KEY'],
                                max_tokens=sb['model_options']['output_level'],
                                tags=[sb['llm_model']])
        elif sb['llm_source'] == 'Hugging Face':
            llm = ChatOpenAI(base_url=sb['hf_endpoint'],
                             model=sb['llm_model'],
                             api_key=secrets['HUGGINGFACEHUB_API_TOKEN'],
                             temperature=sb['model_options']['temperature'],
                             max_tokens=sb['model_options']['output_level'],
                             tags=[sb['llm_model']])
        elif sb['llm_source'] == 'LM Studio (local)':
            # base_url takes local configuration from lm studio, no api key required.
            llm = ChatOpenAI(base_url=sb['llm_model'],
                             temperature=sb['model_options']['temperature'],
                             max_tokens=sb['model_options']['output_level'])
        else:
            raise ValueError("Invalid LLM source specified.")
    elif type == 'rag':   # use for RAG application (summary)
        if sb['rag_llm_source'] == 'OpenAI':
            llm = ChatOpenAI(model_name=sb['rag_llm_model'],
                             openai_api_key=secrets['OPENAI_API_KEY'],
                             temperature=sb['model_options']['temperature'],
                             max_tokens=sb['model_options']['output_level'],
                             tags=[sb['rag_llm_model']])
        elif sb['rag_llm_source'] == 'Hugging Face':
            llm = ChatOpenAI(base_url=sb['rag_hf_endpoint'],
                             model=sb['rag_llm_model'],
                             temperature=sb['model_options']['temperature'],
                             max_tokens=sb['model_options']['output_level'],
                             api_key=secrets['HUGGINGFACEHUB_API_TOKEN'],
                             tags=[sb['rag_llm_model']])
        elif sb['rag_llm_source'] == 'LM Studio (local)':
            # base_url takes local configuration from lm studio, no api key required.
            llm = ChatOpenAI(base_url=sb['rag_llm_model'],
                             model_name='lm_studio')
        else:
            raise ValueError("Invalid LLM source specified.")
    return llm
def get_query_model(sb, secrets):
    """
    Returns the query model based on the provided parameters.

    Args:
        sb (dict): A dictionary containing the parameters for the query model.
        secrets (dict): A dictionary containing the API keys for different query models.

    Returns:
        query_model: The selected query model based on the provided parameters.

    Raises:
        NotImplementedError: If the query model is not recognized.
    """
    if sb['index_type'] == 'RAGatouille':
        query_model = RAGPretrainedModel.from_pretrained(sb['embedding_name'],
                                                         index_root=os.path.join(os.getenv('LOCAL_DB_PATH'),'.ragatouille'))
    elif sb['query_model'] == 'OpenAI':
        query_model = OpenAIEmbeddings(model=sb['embedding_name'], openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['query_model'] == 'Voyage':
        query_model = VoyageAIEmbeddings(model=sb['embedding_name'], voyage_api_key=secrets['VOYAGE_API_KEY'], truncation=False)
    elif sb['query_model'] == 'Hugging Face':
        if sb['embedding_hf_endpoint']:
            query_model = HuggingFaceInferenceAPIEmbeddings(model_name=sb['embedding_name'], 
                                                            api_url=sb['embedding_hf_endpoint'],
                                                            api_key=secrets['HUGGINGFACEHUB_API_TOKEN'])
        else:
            query_model = HuggingFaceInferenceAPIEmbeddings(model_name=sb['embedding_name'], 
                                                            api_key=secrets['HUGGINGFACEHUB_API_TOKEN'])
    else:
        raise NotImplementedError('Query model not recognized.')
    return query_model
def show_pinecone_indexes(format=True):
    """
    Retrieves the list of Pinecone indexes and their status.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        format (bool, optional): Specifies whether to format the output for display.

    Returns:
        dict or str: If format is True, returns a formatted string representation of the Pinecone status.
                    If format is False, returns a dictionary containing the Pinecone status.

    """
    if os.getenv('PINECONE_API_KEY') is None or os.getenv('PINECONE_API_KEY')=='':
        pinecone_status = {'status': False, 'message': 'Pinecone API Key is not set.'}
    else:
        pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        indexes=pc.list_indexes()
        if len(indexes)==0:
            pinecone_status = {'status': False, 'message': 'No indexes found'}
        else:
            pinecone_status = {'status': True, 'message': indexes}
    
    if format:
        return _format_pinecone_status(pinecone_status)
    else:
        return pinecone_status
def show_chroma_collections(format=True):
    """
    Retrieves the list of chroma collections from the local database.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        format (bool, optional): Specifies whether to format the output for display.

    Returns:
        dict or str: If format is True, returns a formatted string representation of the chroma status.
                    If format is False, returns a dictionary containing the chroma status.

    Raises:
        ValueError: If the chroma vector database needs to be reset.

    """
    if os.getenv('LOCAL_DB_PATH') is None or os.getenv('LOCAL_DB_PATH')=='':
        chroma_status = {'status': False, 'message': 'Local database path is not set.'}
    else:
        db_folder_path=os.getenv('LOCAL_DB_PATH')
        try:
            persistent_client = chromadb.PersistentClient(path=os.path.join(db_folder_path,'chromadb'))
        except:
            raise ValueError("Chroma vector database needs to be reset, or the database path is incorrect. Clear cache, or reset path. You may have specified a path which is read only or has no collections.")
        collections=persistent_client.list_collections()
        if len(collections)==0:
            chroma_status = {'status': False, 'message': 'No collections found'}
        else:   
            chroma_status = {'status': True, 'message': collections}
    if format:
        return _format_chroma_status(chroma_status)
    else:
        return chroma_status
def show_ragatouille_indexes(format=True):
    """
    Retrieves the list of ragatouille indexes.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        format (bool, optional): Specifies whether to format the indexes for display.

    Returns:
        dict or str: If format is True, returns a formatted string representation of the ragatouille status.
                    If format is False, returns a dictionary containing the ragatouille status.

    Raises:
        ValueError: If the ragatouille vector database needs to be reset.

    """
    if os.getenv('LOCAL_DB_PATH') is None or os.getenv('LOCAL_DB_PATH')=='':
        ragatouille_status = {'status': False, 'message': 'Local database path is not set.'}
    else:
        db_folder_path=os.getenv('LOCAL_DB_PATH')
        try:
            path=os.path.join(db_folder_path,'.ragatouille/colbert/indexes')
            indexes = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    indexes.append(item)
            if len(indexes)>0:
                ragatouille_status = {'status': True, 'message': indexes}
            else:
                ragatouille_status = {'status': False, 'message': 'No indexes found.'}  # if .ragatouille structure exists but is empty
        except:
            ragatouille_status = {'status': False, 'message': 'No indexes found.'}  # if .ragatouille structure does not exist yet

    if format:
        return _format_ragatouille_status(ragatouille_status)
    else:
        return ragatouille_status
def st_connection_status_expander(expanded: bool = True, delete_buttons: bool = False, set_secrets: bool = False):
    """
    Makes a Streamlit expander widget to display connection status information.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        expanded (bool, optional): Whether the expander is initially expanded or collapsed. Only intended with account access.
        delete_buttons (bool, optional): Whether to display delete buttons for Pinecone and Chroma DB indexes.
        set_secrets (bool, optional): Whether to set the secrets.
    """
    with st.expander("Connection Status", expanded=expanded):
        # Set secrets and assign to environment variables
        if set_secrets:
            st.markdown("**Set API keys**:")
            keys={}
            # OPENAI_API_KEY
            keys['OPENAI_API_KEY'] = st.text_input('OpenAI API Key', type='password',help='OpenAI API Key: https://platform.openai.com/api-keys')
            if keys['OPENAI_API_KEY']!='':
                os.environ['OPENAI_API_KEY'] = keys['OPENAI_API_KEY'] 
            
            # VOYAGE_API_KEY
            keys['VOYAGE_API_KEY'] = st.text_input('Voyage API Key', type='password',help='Voyage API Key: https://dash.voyageai.com/api-keys')
            if keys['VOYAGE_API_KEY']!='':
                os.environ['VOYAGE_API_KEY'] = keys['VOYAGE_API_KEY']
            
            # PINECONE_API_KEY
            keys['PINECONE_API_KEY']=st.text_input('Pinecone API Key',type='password',help='Pinecone API Key: https://www.pinecone.io/')
            if keys['PINECONE_API_KEY']!='':
                os.environ['PINECONE_API_KEY'] = keys['PINECONE_API_KEY']

            # HUGGINGFACEHUB_API_TOKEN
            keys['HUGGINGFACEHUB_API_TOKEN'] = st.text_input('Hugging Face API Key', type='password',help='Hugging Face API Key: https://huggingface.co/settings/tokens')
            if keys['HUGGINGFACEHUB_API_TOKEN']!='':
                os.environ['HUGGINGFACEHUB_API_TOKEN'] = keys['HUGGINGFACEHUB_API_TOKEN']
            
            # LOCAL_DB_PATH
            keys['LOCAL_DB_PATH'] = st.text_input('Update Local Database Path',os.getenv('LOCAL_DB_PATH'),help='Path to local database (e.g. chroma)')
            if keys['LOCAL_DB_PATH']!='':
                os.environ['LOCAL_DB_PATH'] = keys['LOCAL_DB_PATH']

        if os.getenv('LOCAL_DB_PATH') is None:
            raise SecretKeyException('Local Database Path is required. Use an absolute path.','LOCAL_DB_PATH_MISSING')
        else:
            db_folder_path=os.getenv('LOCAL_DB_PATH')

        # Show key status
        st.markdown("**API key status** (Indicates status of local variable. It does not guarantee the key itself is correct):")
        st.markdown(test_key_status())

        # TODO the try statements below sometimes allow delete to be pressed but the index which does exist isn't deleted. Update so an error is thrown if the index is not deleted.
        # Pinecone
        st.markdown(show_pinecone_indexes())
        try:
            pinecone_indexes = [obj.name for obj in show_pinecone_indexes(format=False)['message']]
            if delete_buttons:
                pinecone_index_name = st.selectbox('Pinecone index to delete', pinecone_indexes)
                if st.button('Delete Pinecone index', help='This is permanent!'):
                    if pinecone_index_name.endswith("parent-child"):
                        rag_type = "Parent-Child"
                    elif pinecone_index_name.endswith("summary"):
                        rag_type = "Summary"
                    else:
                        rag_type = "Standard"
                    data_processing.delete_index('Pinecone', pinecone_index_name, rag_type, local_db_path=db_folder_path)
                    st.markdown(f"Index {pinecone_index_name} has been deleted.")
        except:
            pass

        # Chroma DB
        st.markdown(show_chroma_collections())
        try:
            chroma_db_collections = [obj.name for obj in show_chroma_collections(format=False)['message']]
            if delete_buttons:
                chroma_db_name = st.selectbox('Chroma database to delete', chroma_db_collections)
                if st.button('Delete Chroma database', help='This is permanent!'):
                    if chroma_db_name.endswith("parent-child"):
                        rag_type = "Parent-Child"
                    elif "-summary-" in chroma_db_name:
                        rag_type = "Summary"
                    else:
                        rag_type = "Standard"
                    data_processing.delete_index('ChromaDB', chroma_db_name, rag_type, local_db_path=db_folder_path)
                    st.markdown(f"Database {chroma_db_name} has been deleted.")
        except:
            pass
        
        # Ragatouille
        st.markdown(show_ragatouille_indexes())
        try:
            ragatouille_index_data=show_ragatouille_indexes(format=False)
            ragatouille_indexes = [obj for obj in ragatouille_index_data['message']]
            if delete_buttons and ragatouille_index_data['status']==True:
                ragatouille_name = st.selectbox('RAGatouille database to delete', ragatouille_indexes)
                if st.button('Delete RAGatouille database', help='This is permanent!'):
                    data_processing.delete_index('RAGatouille', ragatouille_name, "Standard", local_db_path=db_folder_path)
                    st.markdown(f"Index {ragatouille_name} has been deleted.")
        except:
            pass

        # Local database path
        st.markdown(f"Local database path: {os.environ['LOCAL_DB_PATH']}")
# def st_setup_page(page_title: str, home_dir:str, config_file: str, sidebar_config: dict = None):
#     """
#     Sets up the Streamlit page with the given title and loads the sidebar configuration.

#     Args:
#         page_title (str): The title of the Streamlit page.
#         home_dir (str): The path to the home directory.
#         config_file (str): The path to the configuration file.
#         sidebar_config (dict, optional): The sidebar configuration.

#     Returns:
#         tuple: A tuple containing the following:
#             - paths (dict): A dictionary containing the following directory paths:
#                 - base_folder_path (str): The path to the root folder.
#                 - config_folder_path (str): The path to the config folder.
#                 - data_folder_path (str): The path to the data folder.
#                 - db_folder_path (str): The path to the database folder.
#             - sb (dict): The sidebar configuration.
#             - secrets (dict): A dictionary containing the set API keys.

#     Raises:
#         SecretKeyException: If there is an issue with the secret keys.

#     """
#     # load_dotenv(find_dotenv(), override=True)

#     # base_folder_path = home_dir
#     # data_folder_path=os.path.join(base_folder_path, 'data')

#     # # Set the page title
#     # st.title(page_title)

#     # Set local database
#     # Only show the text input if no value has been entered yet
#     if not os.environ.get('LOCAL_DB_PATH'):
#         local_db_path_input = st.empty()  # Create a placeholder for the text input
#         warn_db_path=st.warning('Local Database Path is required to initialize. Use an absolute path.')
#         local_db_path = local_db_path_input.text_input('Update Local Database Path', help='Path to local database (e.g. chroma).')
#         if local_db_path:
#             os.environ['LOCAL_DB_PATH'] = local_db_path
#         else:
#             st.stop()
#     if os.environ.get('LOCAL_DB_PATH'): # If a value has been entered, update the environment variable and clear the text input
#         try:
#             local_db_path_input.empty()  # This will remove the text input from the page if it exists
#             warn_db_path.empty()
#         except:
#             pass    # If the text input has already been removed, do nothing

#     # # Load sidebar
#     # try:
#     #     if sidebar_config is None:
#     #         sb=load_sidebar(config_file)
#     #     else:
#     #         sb=load_sidebar(config_file,
#     #                         **sidebar_config)
#     # except SecretKeyException as e:
#     #     # If no .env file is found, set the local db path when the warning is raised.
#     #     st.warning(f"{e}")
#     #     st.stop()
#     # try:
#     #     secrets=set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar
#     # except SecretKeyException as e:
#     #     st.warning(f"{e}")
#     #     st.stop()

#     # # Set db folder path based on env variable
#     # db_folder_path=os.environ['LOCAL_DB_PATH']

#     # paths={'base_folder_path':base_folder_path,
#     #        'config_folder_path':config_file,
#     #        'data_folder_path':data_folder_path,
#     #        'db_folder_path':db_folder_path}

#     return paths,sb,secrets
def extract_pages_from_pdf(url, target_page, page_range=5):
    try:
        # Download extracted relevant section of the PDF file
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        pdf_data = response.content

        # Load PDF in PyMuPDF
        doc = fitz.open("pdf", pdf_data)
        extracted_doc = fitz.open()  # New PDF for extracted pages

        # Calculate the range of pages to extract
        start_page = max(target_page, 0)
        end_page = min(target_page + page_range, doc.page_count - 1)

        # Extract specified pages
        for i in range(start_page, end_page + 1):
            extracted_doc.insert_pdf(doc, from_page=i, to_page=i)

        # Save the extracted pages to a new PDF file in memory
        extracted_pdf = extracted_doc.tobytes()
        extracted_doc.close()
        doc.close()
        return extracted_pdf

    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None
    except fitz.FileDataError as e:
        print(f"Error processing PDF: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
def get_pdf(url):
    try:
        # Download full PDF file
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        pdf_data = response.content

        # Load PDF in PyMuPDF
        doc = fitz.open("pdf", pdf_data)

        # Save the extracted pages to a new PDF file in memory
        extracted_pdf = doc.tobytes()
        doc.close()
        return extracted_pdf

    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None
    except fitz.FileDataError as e:
        print(f"Error processing PDF: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
def _format_key_status(key_status: str):
    """
    Formats the key status dictionary into a formatted string.

    Args:
        key_status (str): The dictionary containing the key status information.

    Returns:
        str: The formatted string representing the key status.
    """
    formatted_status = ""
    for key, value in key_status.items():
        status = value['status']
        if status:
            formatted_status += f"- {key}: :white_check_mark:\n"
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
            status = ":white_check_mark:"
            index_description += f"- {name}: {state} ({status})\n"
        markdown_string = f"**Pinecone Indexes**\n{index_description}"
    else:
        message = pinecone_status['message']
        markdown_string = f"**Pinecone Indexes**\n- {message} :x:"
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
            status = ":white_check_mark:"
            collection_description += f"- {name}: ({status})\n"
        markdown_string = f"**ChromaDB Collections**\n{collection_description}"
    else:
        message = chroma_status['message']
        markdown_string = f"**ChromaDB Collections**\n- {message} :x:"
    return markdown_string
def _format_ragatouille_status(ragatouille_status):
    """
    Formats the ragatouille status for display.

    Args:
        ragatouille_status (dict): The ragatouille status dictionary.

    Returns:
        str: A formatted string representation of the ragatouille status.

    """
    index_description=''
    if ragatouille_status['status']:
        for index in ragatouille_status['message']:
            name = index
            status = ":white_check_mark:"
            index_description += f"- {name}: ({status})\n"
        markdown_string = f"**Ragatouille Indexes**\n{index_description}"
    else:
        message = ragatouille_status['message']
        markdown_string = f"**Ragatouille Indexes**\n- {message} :x:"
    return markdown_string
