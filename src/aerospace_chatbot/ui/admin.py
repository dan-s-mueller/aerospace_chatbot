"""Admin interface components."""

import streamlit as st
import os, logging

from ..core.config import load_config, ConfigurationError, get_secrets, set_secrets, get_required_api_keys
# from ..core.cache import Dependencies
from ..services.database import DatabaseService, get_available_indexes

class SidebarManager:
    """Manages the creation, state, and layout of the Streamlit sidebar."""
    
    def __init__(self, config_file):
        self._config_file = config_file
        self._config = load_config(self._config_file)
        # self._deps = Dependencies()
        self.sb_out = st.session_state.get('sb', {})
        if self.sb_out is None:
            self.sb_out = {}
        self._check_local_db_path()
        self.initialize_session_state()
        self.logger = logging.getLogger(__name__)
        
    def initialize_session_state(self):
        """
        Initialize session state for all sidebar elements.
        """
        if 'sidebar_state_initialized' not in st.session_state:
            elements = {
                'index': ['index_selected', 'index_type'],
                'rag': ['rag_llm_service', 'rag_llm_model', 'rag_endpoint'],
                'llm': ['llm_service', 'llm_model', 'llm_endpoint'],
                'model_options': ['temperature', 'output_level', 'k_retrieve', 'k_rerank', 'style_mode'],
                'embeddings': ['embedding_service', 'embedding_model', 'embedding_endpoint'],
                'rerankers': ['rerank_service', 'rerank_model'],
                'api_keys': ['openai_key', 'anthropic_key', 'hf_key', 'voyage_key', 'pinecone_key']
            }
            
            # Get disabled controls from config
            disabled_controls = self._config.get('disabled_controls', {})
            
            for group, group_elements in elements.items():
                for element in group_elements:
                    disabled = False
                    # Check if this element should be disabled based on config
                    if group in disabled_controls and element in disabled_controls[group]:
                        disabled = True
                    
                    if f'{element}_disabled' not in st.session_state:
                        st.session_state[f'{element}_disabled'] = disabled
                    if f'{element}_value' not in st.session_state:
                        st.session_state[f'{element}_value'] = None
            
            st.session_state.sidebar_state_initialized = True
            
    def render_sidebar(self):
        """
        Render the complete sidebar based on enabled options.
        """
        try:
            # Sync any changed values from session state first
            for key in ['index_type', 'embedding_model', 'embedding_service', 
                       'openai_key', 'anthropic_key', 'voyage_key', 'pinecone_key', 'hf_key']:
                if f'{key}_select' in st.session_state:
                    self.sb_out[key] = st.session_state[f'{key}_select']
            
            # Initialize remaining dependencies
            self._ensure_dependencies()

            # Log the current state before rendering
            self.logger.info(f"Starting sidebar render with settings: {self.sb_out}")

            # Render GUI elements
            self.available_indexes, self.index_metadatas = self._render_index_selection()
            self._render_llm()
            self._render_model_options()
            self._render_vector_database()
            self._render_embeddings()
            self._render_rerankers()
            self._render_secret_keys()

            # Update session state with final values
            st.session_state.sb = self.sb_out
            return self.sb_out
        except ConfigurationError as e:
            st.error(f"Configuration error: {e}")
            st.stop()
        
    def _ensure_dependencies(self):
        """Pre-initialize all required dependencies before rendering GUI"""
        # Handle core dependencies first
        if 'index_type' not in self.sb_out:
            # Get first database name from the list
            self.sb_out['index_type'] = self._config['databases'][0]['name']    # Default to first database type in config
        
        if 'embedding_service' not in self.sb_out:
            # Find the database config by name
            db_config = next(db for db in self._config['databases'] if db['name'] == self.sb_out['index_type'])
            embedding_service = db_config['embedding_services'][0]    # Default to first embedding service in config
            self.sb_out['embedding_service'] = embedding_service
            
            if self.sb_out['index_type'] == 'RAGatouille':
                self.sb_out['embedding_model'] = embedding_service
            else:
                # Find the embedding config by name
                embedding_config = next(e for e in self._config['embeddings'] if e['service'] == embedding_service)
                self.sb_out['embedding_model'] = embedding_config['models'][0]  # Default to first embedding model in config
            
            self.logger.info(f"Embedding service: {self.sb_out['embedding_service']}, embedding model: {self.sb_out['embedding_model']}")

        # Validate required API keys based on config
        secrets = get_secrets()
        selected_services = {
            'index_type': self.sb_out.get('index_type'),
            'embedding_service': self.sb_out.get('embedding_service'),
            'llm_service': self.sb_out.get('llm_service')
        }
        
        required_keys = get_required_api_keys(self._config, selected_services)
        
        for service, env_var in required_keys.items():
            if env_var not in secrets or not secrets[env_var]:
                raise ConfigurationError(f"{service} API key ({env_var}) is required for the selected configuration")

    def _render_index_selection(self):
        """Render index selection section."""
        st.sidebar.title('Index Selected')

        # Check if the configuration is set to 'tester'
        if os.getenv('AEROSPACE_CHATBOT_CONFIG') == 'tester':
            # Use the index from the config and disable the selection
            tester_config = self._config.get('available_indexes', [])
            if tester_config:
                self.sb_out['index_selected'] = tester_config[0]
                st.session_state.index_selected_disabled = True
                st.sidebar.selectbox(
                    'Index selected',
                    tester_config,
                    disabled=True,
                    help='Index is pre-selected for tester configuration.',
                    key='index_selected_select'
                )
                self.logger.info(f"Index selected: {self.sb_out['index_selected']}")
                return tester_config, {}  # Return the pre-selected index
        else:
            # Original code for rendering index selection
            self.logger.info(f"Getting available indexes for sidebar with settings: {self.sb_out}")
            available_indexes, index_metadatas = get_available_indexes(
                self.sb_out['index_type'],
                self.sb_out['embedding_model'],
            )
            self.logger.info(f"Available indexes for sidebar: {available_indexes}")

            if not available_indexes:
                st.warning('No compatible collections found with current settings.')
                return [], {}  # Return empty lists instead of None

            # Only show indexes that match the current settings
            self.sb_out['index_selected'] = st.sidebar.selectbox(
                'Index selected',
                available_indexes,
                disabled=st.session_state.index_selected_disabled,
                help='Select the index to use for the application.',
                key='index_selected_select'
            )

            self.logger.info(f"Index selected: {self.sb_out['index_selected']}")
            return available_indexes, index_metadatas
        
    def _render_llm(self):
        """Render LLM selection section."""
        st.sidebar.title('Large Language Model')
        
        llm_services = [llm['service'] for llm in self._config['llms']]
        self.sb_out['llm_service'] = st.sidebar.selectbox(
            'LLM Service',
            llm_services,
            disabled=st.session_state.llm_service_disabled
        )
        
        # Find the LLM config by name
        llm_config = next(llm for llm in self._config['llms'] if llm['service'] == self.sb_out['llm_service'])
        models = llm_config.get('models', [])  # Use get() in case 'models' key doesn't exist
        self.sb_out['llm_model'] = st.sidebar.selectbox(
            'LLM Model',
            models,
            disabled=st.session_state.llm_model_disabled
        )
        self.logger.info(f"LLM service: {self.sb_out['llm_service']}, LLM model: {self.sb_out['llm_model']}")

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
            value=5000,
            disabled=st.session_state.output_level_disabled,
            help='Max output tokens for LLM. Concise: 50, Verbose: 1000+. Limit depends on model.'
        )

        style_mode = st.sidebar.selectbox(
            'Style mode',
            self._config['style_modes'],
            disabled=st.session_state.style_mode_disabled,
            help='Select the style mode for the chatbot. Standard is the default, which is neutral and informative. All styles will remain factual and accurate.'
        )
        
        st.sidebar.title('Retrieval Options')
        k_retrieve = st.sidebar.number_input(
            'Number of chunks retrieved per prompt',
            min_value=1,
            step=1,
            value=20,
            disabled=st.session_state.k_retrieve_disabled,
            help='Number of items to retrieve per query.'
        )

        k_rerank = st.sidebar.number_input(
            'Number of docs reranked to use for response',
            min_value=1,
            step=1,
            value=5,
            disabled=st.session_state.k_rerank_disabled,
            help='Number of items to rerank from retrieved chunks. This many chunks are ranked by relevance and provided to the LLM for a response. Must be at least k_retrieve.'
        )
        
        if self.sb_out['index_type'] != 'RAGatouille':
            self.sb_out['model_options'] = {
                'output_level': output_level,
                'style_mode': style_mode,
                'k_retrieve': k_retrieve,
                'k_rerank': k_rerank,
                'temperature': temperature,
            }
        else:
            self.sb_out['model_options'] = {
                'output_level': output_level,
                'style_mode': style_mode,
                'k_retrieve': k_retrieve,
                'temperature': temperature,
            }

        self.logger.info(f"Model options: {self.sb_out['model_options']}")

    def _render_vector_database(self):
        """Render vector database section."""
        st.sidebar.title('Vector Database Type')
        
        # Get list of database names from the config list
        database_names = [db['name'] for db in self._config['databases']]
        
        self.sb_out['index_type'] = st.sidebar.selectbox(
            'Index type',
            database_names,
            disabled=st.session_state.index_type_disabled,
            help='Select the type of index to use.'
        )

        self.logger.info(f"Index type: {self.sb_out['index_type']}")

    def _render_embeddings(self):
        """
        Render embeddings configuration section.
        """
        st.sidebar.title('Embeddings', 
                        help='See embedding leaderboard here for performance overview: https://huggingface.co/spaces/mteb/leaderboard')
        
        # Find the database config for the selected index type
        db_config = next(db for db in self._config['databases'] if db['name'] == self.sb_out['index_type'])
        
        if self.sb_out['index_type'] == 'RAGatouille':
            # For RAGatouille, use embedding_services directly as models
            self.sb_out['embedding_model'] = st.sidebar.selectbox(
                'Embedding model',
                db_config['embedding_services'],  # Use embedding_services instead of embedding_models
                disabled=st.session_state.embedding_model_disabled,
                help="Models listed are compatible with the selected index type.",
                key='embedding_model_select'
            )
            self.sb_out['embedding_service'] = 'RAGatouille'  # Set a default service for RAGatouille
        else:
            self.sb_out['embedding_service'] = st.sidebar.selectbox(
                'Embedding service',
                db_config['embedding_services'],
                disabled=st.session_state.embedding_service_disabled,
                help="Model provider.",
                key='embedding_service_select'
            )
            
            # Find the embedding config for the selected model
            embedding_config = next(e for e in self._config['embeddings'] if e['service'] == self.sb_out['embedding_service'])
            
            self.sb_out['embedding_model'] = st.sidebar.selectbox(
                'Embedding model',
                embedding_config['models'],
                disabled=st.session_state.embedding_model_disabled,
                help="Models listed are compatible with the selected index type.",
                key='embedding_model_select'
            )
            
            if self.sb_out['embedding_model'] == "Dedicated Endpoint":
                self.sb_out['embedding_hf_endpoint'] = st.sidebar.text_input(
                    'Dedicated endpoint URL',
                    '',
                    disabled=st.session_state.embedding_endpoint_disabled,
                    help='See Hugging Face configuration for endpoint.'
                )
            else:
                self.sb_out['embedding_hf_endpoint'] = None

        self.logger.info(f"Embedding service: {self.sb_out['embedding_service']}, embedding model: {self.sb_out['embedding_model']}")

    def _render_rerankers(self):
        """
        Render rerankers configuration section.
        """
        st.sidebar.title('Rerankers')

        rerank_services = [reranker["service"] for reranker in self._config['rerankers']]

        self.sb_out['rerank_service'] = st.sidebar.selectbox(
            'Reranker service',
            rerank_services,
            disabled=st.session_state.rerank_service_disabled
        )

        # Find the reranker config for the selected model
        reranker_config = next(e for e in self._config['rerankers'] if e['service'] == self.sb_out['rerank_service'])
        
        self.sb_out['rerank_model'] = st.sidebar.selectbox(
            'Reranker model',
            reranker_config['models'],
            disabled=st.session_state.rerank_model_disabled
        )

        self.logger.info(f"Reranker service: {self.sb_out['rerank_service']}, reranker model: {self.sb_out['rerank_model']}")

    def _render_secret_keys(self):
        """
        Render secret keys configuration section.
        """
        self.sb_out['keys'] = {}
        st.sidebar.title('Secret keys', help='See Home page under Connection Status for status of keys.')
        
        # Get existing secrets from environment
        existing_secrets = get_secrets()
        
        # Only show relevant API key inputs based on selected models
        if ('llm_service' in self.sb_out and self.sb_out['llm_service'] == 'OpenAI') or \
           ('embedding_service' in self.sb_out and self.sb_out['embedding_service'] == 'OpenAI'):
            key = 'OPENAI_API_KEY'
            self.sb_out['keys'][key] = st.sidebar.text_input(
                'OpenAI API Key',
                value='.env' if key in existing_secrets else '',
                disabled=key in existing_secrets or st.session_state.openai_key_disabled,
                type='default' if key in existing_secrets else 'password',
                help='OpenAI API Key: https://platform.openai.com/api-keys',
                key='openai_key_select'  # Add unique key for state management
            )
        
        if 'llm_service' in self.sb_out and self.sb_out['llm_service'] == 'Anthropic':
            key = 'ANTHROPIC_API_KEY'
            self.sb_out['keys'][key] = st.sidebar.text_input(
                'Anthropic API Key',
                value='.env' if key in existing_secrets else '',
                disabled=key in existing_secrets or st.session_state.anthropic_key_disabled,
                type='default' if key in existing_secrets else 'password',
                help='Anthropic API Key: https://console.anthropic.com/settings/keys',
                key='anthropic_key_select'
            )
        
        if 'embedding_service' in self.sb_out and self.sb_out['embedding_service'] == 'Voyage':
            key = 'VOYAGE_API_KEY'
            self.sb_out['keys'][key] = st.sidebar.text_input(
                'Voyage API Key',
                value='.env' if key in existing_secrets else '',
                disabled=key in existing_secrets or st.session_state.voyage_key_disabled,
                type='default' if key in existing_secrets else 'password',
                help='Voyage API Key: https://dash.voyageai.com/api-keys',
                key='voyage_key_select'
            )
        
        if 'index_type' in self.sb_out and self.sb_out['index_type'] == 'Pinecone':
            key = 'PINECONE_API_KEY'
            self.sb_out['keys'][key] = st.sidebar.text_input(
                'Pinecone API Key',
                value='.env' if key in existing_secrets else '',
                disabled=key in existing_secrets or st.session_state.pinecone_key_disabled,
                type='default' if key in existing_secrets else 'password',
                help='Pinecone API Key: https://www.pinecone.io/',
                key='pinecone_key_select'
            )
        
        if ('llm_service' in self.sb_out and self.sb_out['llm_service'] == 'Hugging Face') or \
           ('embedding_service' in self.sb_out and self.sb_out['embedding_service'] == 'Hugging Face'):
            key = 'HUGGINGFACEHUB_API_KEY'
            self.sb_out['keys'][key] = st.sidebar.text_input(
                'Hugging Face API Key',
                value='.env' if key in existing_secrets else '',
                disabled=key in existing_secrets or st.session_state.hf_key_disabled,
                type='default' if key in existing_secrets else 'password',
                help='Hugging Face API Key: https://huggingface.co/settings/tokens',
                key='hf_key_select'
            )
        
        # Update session state immediately with any changes
        st.session_state.sb = self.sb_out
        
        # Set secrets and environment variables
        try:
            secrets = get_secrets()  # Get secrets from environment
            self.secrets = set_secrets(secrets)  # Set them as environment variables
        except ConfigurationError as e:
            st.warning(f"{e}")
            st.stop()

    def _check_local_db_path(self):
        """
        Check and set local database path if not already set.
        """
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