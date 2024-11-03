from .core import Dependencies, load_config, get_cache_decorator, get_cache_data_decorator
from .services import (
    DatabaseService, 
    EmbeddingService, 
    LLMService,
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
    DEFAULT_DOCUMENT_PROMPT,
    GENERATE_SIMILAR_QUESTIONS,
    CLUSTER_LABEL
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
    show_connection_status
)

__version__ = "0.0.9"

__all__ = [
    # Core
    'Dependencies',
    'load_config',
    'get_cache_decorator',
    'get_cache_data_decorator',
    
    # Services
    'DatabaseService',
    'EmbeddingService',
    'LLMService',
    'CONDENSE_QUESTION_PROMPT',
    'QA_PROMPT',
    'DEFAULT_DOCUMENT_PROMPT',
    'GENERATE_SIMILAR_QUESTIONS',
    'CLUSTER_LABEL',
    
    # Processing
    'DocumentProcessor',
    'QAModel',
    'ChunkingResult',
    
    # UI
    'SidebarManager',
    'display_chat_history',
    'display_sources',
    'setup_page_config',
    'show_connection_status'
]

def get_version():
    """Return the current version of the package."""
    return __version__
