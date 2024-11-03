"""Admin interface components."""

import streamlit as st
from typing import Dict, Optional
from pathlib import Path

from ..core.cache import Dependencies
from ..core.config import load_config, ConfigurationError
from ..services.database import DatabaseService
from ..services.embeddings import EmbeddingService
from ..services.llm import LLMService

class SidebarManager:
    """Manages the creation, state, and layout of the Streamlit sidebar."""
    
    def __init__(self, config_file):
        self._config_file = config_file
        self._config = None
        self._deps = Dependencies()
        self.sb_out = {}
        self._check_local_db_path()
        self._load_config()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state for all sidebar elements."""
        if 'sidebar_state_initialized' not in st.session_state:
            elements = {
                'index': ['index_selected', 'index_type'],
                'embeddings': ['query_model', 'embedding_name', 'embedding_endpoint'],
                'rag': ['rag_type', 'rag_llm_source', 'rag_llm_model', 'rag_endpoint'],
                'llm': ['llm_source', 'llm_model', 'llm_endpoint'],
                'model_options': ['temperature', 'output_level', 'k', 'search_type'],
                'api_keys': ['openai_key', 'anthropic_key', 'hf_key', 'voyage_key', 'pinecone_key']
            }
            
            # Get disabled controls from config
            disabled_controls = self._config.get('disabled_controls', {})
            
            for group, group_elements in elements.items():
                for element in group_elements:
                    disabled = element in disabled_controls.get(group, [])
                    if f'{element}_disabled' not in st.session_state:
                        st.session_state[f'{element}_disabled'] = disabled
                    if f'{element}_value' not in st.session_state:
                        st.session_state[f'{element}_value'] = None
            
            st.session_state.sidebar_state_initialized = True
            
    def render_sidebar(self):
        """Render the complete sidebar based on enabled options."""
        try:
            # Initialize all dependencies before rendering
            self._ensure_dependencies()

            # Render GUI elements
            self._render_index_selection()
            self._render_llm()
            self._render_rag_type()    
            self._render_model_options()
            self._render_vector_database()
            self._render_embeddings()
            self._render_secret_keys()

        except ConfigurationError as e:
            st.error(f"Configuration error: {e}")
            st.stop()
            
        return self.sb_out
        
    def _render_index_selection(self):
        """Render index selection section."""
        st.sidebar.title('Index Selected')
        
        # Get available indexes based on current settings
        available_indexes = self._get_available_indexes()
        
        if not available_indexes:
            st.warning('No compatible collections found.')
            return
            
        self.sb_out['index_selected'] = st.sidebar.selectbox(
            'Index selected', 
            available_indexes,
            disabled=st.session_state.index_selected_disabled,
            help='Select the index to use for the application.'
        )
        
    def _render_llm(self):
        """Render LLM selection section."""
        st.sidebar.title('Language Model')
        
        llm_sources = self._config['llms'].keys()
        self.sb_out['llm_source'] = st.sidebar.selectbox(
            'LLM Provider',
            llm_sources,
            disabled=st.session_state.llm_source_disabled
        )
        
        models = self._config['llms'][self.sb_out['llm_source']]['models']
        self.sb_out['llm_model'] = st.sidebar.selectbox(
            'Model',
            models,
            disabled=st.session_state.llm_model_disabled
        )
        
    def _render_rag_type(self):
        """Render RAG type configuration section."""
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
                self._config['rag_types'],
                disabled=st.session_state.rag_type_disabled,
                help='Parent-Child is for parent-child RAG. Summary is for summarization RAG.'
            )
            
            if self.sb_out['rag_type'] == 'Summary':
                self._render_rag_llm_config()

    def _render_rag_llm_config(self):
        """Render RAG LLM configuration section."""
        self.sb_out['rag_llm_source'] = st.sidebar.selectbox(
            'RAG LLM model',
            list(self._config['llms'].keys()),
            disabled=st.session_state.rag_llm_source_disabled,
            help='Select the LLM model for RAG.'
        )
        
        self._render_llm_model_selection('rag_')

    def _render_model_options(self):
        """Render model options section."""
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

    def _render_vector_database(self):
        """Render vector database section."""
        st.sidebar.title('Vector Database Type')
        self.sb_out['index_type'] = st.sidebar.selectbox(
            'Index type',
            list(self._config['databases'].keys()),
            disabled=st.session_state.index_type_disabled,
            help='Select the type of index to use.'
        )

    def _render_embeddings(self):
        """Render embeddings configuration section."""
        st.sidebar.title('Embeddings', 
                        help='See embedding leaderboard here for performance overview: https://huggingface.co/spaces/mteb/leaderboard')
        
        if self.sb_out['index_type'] == 'RAGatouille':
            self.sb_out['query_model'] = st.sidebar.selectbox(
                'Hugging face rag models',
                self._config['databases'][self.sb_out['index_type']]['embedding_models'],
                disabled=st.session_state.query_model_disabled,
                help="Models listed are compatible with the selected index type."
            )
            self.sb_out['embedding_name'] = self.sb_out['query_model']
        else:
            self.sb_out['query_model'] = st.sidebar.selectbox(
                'Embedding model family',
                self._config['databases'][self.sb_out['index_type']]['embedding_models'],
                disabled=st.session_state.query_model_disabled,
                help="Model provider."
            )
            self.sb_out['embedding_name'] = st.sidebar.selectbox(
                'Embedding model',
                self._config['embeddings'][self.sb_out['query_model']]['embedding_models'],
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

    def _render_secret_keys(self):
        """Render secret keys configuration section."""
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
