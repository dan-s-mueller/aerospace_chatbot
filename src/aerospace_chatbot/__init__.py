from .core import (
    Dependencies, 
    load_config, 
    get_cache_decorator, 
    get_cache_data_decorator,
    ConfigurationError,
    ensure_paths,
    cache_resource
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
from .processing import (
    DocumentProcessor, 
    QAModel,
    ChunkingResult
)
from .ui import (
    SidebarManager, 
    display_chat_history, 
    display_sources, 
    setup_page_config,
    show_connection_status,
    get_or_create_spotlight_viewer,
    extract_pages_from_pdf,
    get_pdf
)

__version__ = "0.0.9"

__all__ = [
    # Core
    'Dependencies',
    'load_config',
    'get_cache_decorator',
    'get_cache_data_decorator',
    'ConfigurationError',
    'ensure_paths',
    'cache_resource',
    
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
    'display_chat_history',
    'display_sources',
    'setup_page_config',
    'show_connection_status',
    'get_or_create_spotlight_viewer',
    'extract_pages_from_pdf',
    'get_pdf'
]

def get_version():
    """Return the current version of the package."""
    return __version__
