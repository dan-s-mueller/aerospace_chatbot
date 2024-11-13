from .core import (
    Dependencies, 
    ConfigurationError,
    load_config,
    get_secrets,
    set_secrets,
    setup_logging
)
from .processing import (
    DocumentProcessor, 
    QAModel,
    ChunkingResult
)
from .services import (
    DatabaseService, 
    get_docs_questions_df, 
    add_clusters, 
    export_to_hf_dataset, 
    get_database_status,
    get_available_indexes,
    EmbeddingService, 
    LLMService,
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
    DEFAULT_DOCUMENT_PROMPT,
    GENERATE_SIMILAR_QUESTIONS,
    GENERATE_SIMILAR_QUESTIONS_W_CONTEXT,
    CLUSTER_LABEL,
    SUMMARIZE_TEXT
)
from .ui import (
    SidebarManager,
    setup_page_config,
    display_chat_history, 
    display_sources, 
    show_connection_status,
    handle_file_upload,
    get_or_create_spotlight_viewer
)

__version__ = "0.0.9"

__all__ = [
    # Core
    'Dependencies', 
    'ConfigurationError',
    'load_config',
    'get_secrets',
    'set_secrets',  
    'setup_logging',
    # Services
    'DatabaseService',
    'get_docs_questions_df', 
    'add_clusters', 
    'export_to_hf_dataset', 
    'get_database_status',
    'get_available_indexes',
    'EmbeddingService',
    'LLMService',
    'CONDENSE_QUESTION_PROMPT',
    'QA_PROMPT',
    'DEFAULT_DOCUMENT_PROMPT',
    'GENERATE_SIMILAR_QUESTIONS',
    'GENERATE_SIMILAR_QUESTIONS_W_CONTEXT',
    'CLUSTER_LABEL',
    'SUMMARIZE_TEXT',
    
    # Processing
    'DocumentProcessor',
    'QAModel',
    'ChunkingResult',
    
    # UI
    'SidebarManager',
    'setup_page_config',
    'display_chat_history', 
    'display_sources', 
    'show_connection_status',
    'handle_file_upload',
    'get_or_create_spotlight_viewer'
]

def get_version():
    """Return the current version of the package."""
    return __version__
