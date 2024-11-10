"""Service layer for database, embeddings, and LLM interactions."""

from .database import (
    DatabaseService, 
    get_docs_questions_df, 
    add_clusters, 
    export_to_hf_dataset, 
    get_database_status,
    get_available_indexes
)
from .embeddings import EmbeddingService
from .llm import LLMService
from .prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
    CLUSTER_LABEL,
    TEST_QUERY_PROMPT,
    DEFAULT_DOCUMENT_PROMPT,
    GENERATE_SIMILAR_QUESTIONS,
    GENERATE_SIMILAR_QUESTIONS_W_CONTEXT,
    SUMMARIZE_TEXT
)

__all__ = [
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
    'CLUSTER_LABEL',
    'TEST_QUERY_PROMPT',
    'DEFAULT_DOCUMENT_PROMPT',
    'GENERATE_SIMILAR_QUESTIONS',
    'GENERATE_SIMILAR_QUESTIONS_W_CONTEXT',
    'SUMMARIZE_TEXT'
]
