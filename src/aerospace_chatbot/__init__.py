from .core import (
    ConfigurationError,
    load_config,
    get_secrets,
    set_secrets,
    setup_logging,
    get_required_api_keys
)
from .processing import (
    ChunkingResult,
    DocumentProcessor, 
    QAModel
)
from .services import (
    DatabaseService, 
    get_docs_df, 
    add_clusters, 
    export_to_hf_dataset, 
    get_database_status,
    get_available_indexes,
    EmbeddingService, 
    RerankService,
    LLMService,
    InLineCitationsResponse,
    AltQuestionsResponse,
    style_mode,
    CHATBOT_SYSTEM_PROMPT,
    QA_PROMPT,
    SUMMARIZE_TEXT,
    GENERATE_SIMILAR_QUESTIONS_W_CONTEXT,
    CLUSTER_LABEL,
    DEFAULT_DOCUMENT_PROMPT,
)
from .ui import (
    SidebarManager,
    setup_page_config,
    handle_sidebar_state,
    display_sources,
    process_source_documents,
    show_connection_status,
    handle_file_upload
)

__version__ = "0.0.9"

__all__ = [
    # Core
    'Dependencies', 
    'cache_data',
    'cache_resource',
    'ConfigurationError',
    'load_config',
    'get_secrets',
    'set_secrets',  
    'setup_logging',
    'get_required_api_keys',

    # Services
    'DatabaseService',
    'get_docs_df', 
    'add_clusters', 
    'export_to_hf_dataset', 
    'get_database_status',
    'get_available_indexes',
    'EmbeddingService',
    'RerankService',
    'LLMService',
    'InLineCitationsResponse',
    'AltQuestionsResponse',
    'style_mode',
    'CHATBOT_SYSTEM_PROMPT',
    'QA_PROMPT',
    'SUMMARIZE_TEXT',
    'GENERATE_SIMILAR_QUESTIONS_W_CONTEXT',
    'CLUSTER_LABEL',
    'DEFAULT_DOCUMENT_PROMPT',
    
    # Processing
    'ChunkingResult',
    'DocumentProcessor',
    'QAModel',
    
    # UI
    'SidebarManager',
    'setup_page_config',
    'handle_sidebar_state',
    'display_sources',
    'process_source_documents',
    'show_connection_status',
    'handle_file_upload'
]

def get_version():
    """Return the current version of the package."""
    return __version__
