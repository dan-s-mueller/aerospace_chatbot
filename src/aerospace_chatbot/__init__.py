from .core import (
    Dependencies, 
    cache_resource,
    ConfigurationError,
    get_cache_decorator, 
    get_cache_data_decorator,
    load_config,
    get_secrets
)
from .processing import (
    DocumentProcessor, 
    QAModel,
    ChunkingResult
)
from .services import (
    DatabaseService, 
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
    'cache_resource',
    'ConfigurationError',
    'get_cache_decorator', 
    'get_cache_data_decorator',
    'load_config',
    'get_secrets',
    
    # Services
    'DatabaseService',
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
