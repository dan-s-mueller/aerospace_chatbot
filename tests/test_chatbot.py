import os, sys, json
import itertools
import pytest
import pandas as pd
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from pinecone import Pinecone as pinecone_client
import chromadb
from ragatouille import RAGPretrainedModel

from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

from aerospace_chatbot.core import (
    Dependencies, 
    ConfigurationError,
    load_config,
    get_secrets,
    set_secrets
)
from aerospace_chatbot.processing import (
    DocumentProcessor, 
    QAModel,
    ChunkingResult
)
from aerospace_chatbot.services import (
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
from aerospace_chatbot.ui import (
    SidebarManager,
    setup_page_config,
    display_chat_history, 
    display_sources, 
    show_connection_status,
    handle_file_upload,
    get_or_create_spotlight_viewer
)

# Import local variables
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src/aerospace_chatbot'))

# High priority updates:
# FIXME test that prompts are formatted properly
# FIXME add test for dependency caching
# FIXME add upload file test

# Low priority updates:
# TODO add tests to check conversation history functionality
# TODO add a test to check parent/child and summary lookup functionality (not just that it executes)
# TODO add apptest from streamlit


# Functions
def permute_tests(test_data):
    '''Generate permutations of test cases.'''
    rows = []
    idx = 0
    for row_data in test_data:
        keys = row_data.keys()
        values = row_data.values()
        permutations = list(itertools.product(*values))
        for perm in permutations:
            row = dict(zip(keys, perm))
            row['id'] = idx
            rows.append(row)
            idx += 1
    return rows
def read_test_cases(json_path: str):
    with open(json_path, 'r') as json_file:
        test_cases = json.load(json_file)
    return test_cases
def pytest_generate_tests(metafunc):
    '''
    Use pytest_generate_tests to dynamically generate tests.
    Tests generates tests from a static file (test_cases.json). See test_cases.json for more details.
    '''
    if 'test_input' in metafunc.fixturenames:
        tests = read_test_cases(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_cases.json'))
        metafunc.parametrize('test_input', tests)
def parse_test_case(setup, test_case):
    ''' Parse test case to be used in the test functions.'''
    parsed_test = {
        'id': test_case['id'],
        'index_type': setup['index_type'][test_case['index_type']],
        'embedding_family': test_case['embedding_family'],
        'embedding_name': test_case['embedding_name'],
        'rag_type': setup['rag_type'][test_case['rag_type']],
        'llm_family': test_case['llm_family'],
        'llm': test_case['llm']
    }
    print_str = ', '.join(f'{key}: {value}' for key, value in test_case.items())

    return parsed_test, print_str

# Fixtures
@pytest.fixture(autouse=True)
def setup_fixture():
    """Sets up the fixture for testing the backend."""
    # Override environment variables first, before loading .env
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['LOG_FILE'] = 'logs/test_chatbot.log'
    
    # Setup logging with overridden values
    from aerospace_chatbot.core.config import setup_logging
    logger = setup_logging()
    
    # Now load .env file (won't override existing environment variables)
    load_dotenv(find_dotenv(), override=False)
    
    # Pull api keys from .env file
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')
    HUGGINGFACEHUB_API_KEY = os.getenv('HUGGINGFACEHUB_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')    
    
    # Set environment variables from .env file. They are required for items tested here. This is done in the GUI setup.
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    os.environ['VOYAGE_API_KEY'] = VOYAGE_API_KEY
    os.environ['HUGGINGFACEHUB_API_KEY'] = HUGGINGFACEHUB_API_KEY
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

    LOCAL_DB_PATH = os.path.abspath(os.path.dirname(__file__))   # Default to the test path for easy cleanup.
    # Set default to environment variable
    os.environ['LOCAL_DB_PATH'] = LOCAL_DB_PATH
    
    # Fixed inputs
    docs = ['test1.pdf', 'test2.pdf']
    for i in range(len(docs)):
        docs[i] = os.path.join(os.path.abspath(os.path.dirname(__file__)), docs[i])

    chunk_method = 'character_recursive'
    chunk_size = 400
    chunk_overlap = 0
    batch_size = 50
    test_prompt = 'What are some nuances associated with the analysis and design of hinged booms?'   # Info on test2.pdf

    # Variable inputs
    index_type = {index: index for index in ['ChromaDB', 'Pinecone', 'RAGatouille']}
    rag_type = {rag: rag for rag in ['Standard', 'Parent-Child', 'Summary']}
    
    mock_embedding_service = EmbeddingService(
        model_name='text-embedding-3-small',
        model_type='OpenAI'
    )

    mock_llm_service = LLMService(
        model_name='gpt-4o-mini',
        model_type='OpenAI',
        temperature=0,
        max_tokens=1000
    )

    setup = {
        'logger': logger,
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'ANTHROPIC_API_KEY': ANTHROPIC_API_KEY,
        'VOYAGE_API_KEY': VOYAGE_API_KEY,
        'HUGGINGFACEHUB_API_KEY': HUGGINGFACEHUB_API_KEY,
        'PINECONE_API_KEY': PINECONE_API_KEY,
        'LOCAL_DB_PATH': LOCAL_DB_PATH,
        'docs': docs,
        'chunk_method': chunk_method,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'batch_size': batch_size,
        'test_prompt': test_prompt,
        'index_type': index_type,
        'rag_type': rag_type,
        # Add mock services
        'mock_embedding_service': mock_embedding_service,
        'mock_llm_service': mock_llm_service
    }

    return setup
def test_validate_index(setup_fixture):
    """Test edge cases for validate_index function."""
    from aerospace_chatbot.services.database import DatabaseService
    from aerospace_chatbot.processing import DocumentProcessor
    
    logger = setup_fixture['logger']
    logger.info("Starting validate_index test")

    # Test case 1: Empty index name
    db_type='ChromaDB'
    db_service = DatabaseService(
        db_type=db_type,
        index_name='',
        rag_type='Standard',
        embedding_service=setup_fixture['mock_embedding_service'],
        doc_type='document'
    )

    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        rag_type='Standard',
        chunk_size=400,
        chunk_overlap=50
    )
    
    with pytest.raises(ValueError, match="Index name cannot be empty"):
        db_service.index_name = ""
        db_service._validate_index()

    # Test case 2: Whitespace-only index name
    with pytest.raises(ValueError, match="Index name cannot be empty"):
        db_service.index_name = "   "
        db_service._validate_index()

    # Test case 3: ChromaDB invalid characters
    with pytest.raises(ValueError, match="can only contain alphanumeric characters"):
        db_service.index_name = "test@index"
        db_service._validate_index()

    # Test case 4: ChromaDB consecutive periods
    with pytest.raises(ValueError, match="can only contain alphanumeric characters, underscores, or hyphens"):
        db_service.index_name = "test..index"
        db_service._validate_index()

    # Test case 5: ChromaDB non-alphanumeric start/end
    with pytest.raises(ValueError, match="must start and end with an alphanumeric character"):
        db_service.index_name = "-testindex-"
        db_service._validate_index()

    # Test case 6: ChromaDB name too long
    with pytest.raises(ValueError, match="must be less than 63 characters"):
        db_service.index_name = "a" * 64
        db_service._validate_index()

    # Test case 7: Pinecone name too long
    db_type='Pinecone'
    db_service = DatabaseService(
        db_type=db_type,
        index_name='',
        rag_type='Standard',
        embedding_service=setup_fixture['mock_embedding_service'],
        doc_type='document'
    )

    with pytest.raises(ValueError, match="must be less than 45 characters"):
        db_service.index_name = "a" * 46
        db_service._validate_index()

    # Test case 9: Valid cases with different RAG types
    index_name = 'test-index'
    db_service = DatabaseService(
        db_type='ChromaDB',
        index_name=index_name,
        rag_type='Standard',
        embedding_service=setup_fixture['mock_embedding_service'],
        doc_type='document'
    )

    # Standard RAG
    db_service._validate_index()
    assert db_service.index_name == index_name
    
    # Parent-Child RAG
    db_service.index_name = index_name  
    db_service.rag_type = 'Parent-Child'
    db_service._validate_index()
    assert db_service.index_name == index_name + "-parent-child"
    
    # Summary RAG
    db_service.index_name = index_name
    db_service.rag_type = 'Summary'
    db_service.llm_service = setup_fixture['mock_llm_service']
    db_service._validate_index()
    assert db_service.index_name == index_name + "-summary"
def test_process_documents_standard(setup_fixture):
    '''Test document processing with standard RAG.'''
    logger = setup_fixture['logger']
    logger.info("Starting process_documents_standard test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        rag_type=setup_fixture['rag_type']['Standard'],
        chunk_method=setup_fixture['chunk_method'],
        chunk_size=setup_fixture['chunk_size'],
        chunk_overlap=setup_fixture['chunk_overlap']
    )
    
    result = doc_processor.process_documents(setup_fixture['docs'])
    
    page_ids = [DocumentProcessor.stable_hash_meta(page.metadata) for page in result.pages]
    chunk_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in result.chunks]
    
    assert result.rag_type == setup_fixture['rag_type']['Standard']
    assert result.pages is not None
    assert len(page_ids) == len(set(page_ids))
    assert result.chunks is not None
    assert len(chunk_ids) == len(set(chunk_ids))
    assert result.splitters is not None
def test_process_docs_merge_nochunk(setup_fixture):
    """Test case for document processing with no chunking and merging."""
    logger = setup_fixture['logger']
    logger.info("Starting process_docs_merge_nochunk test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        rag_type=setup_fixture['rag_type']['Standard'],
        chunk_method='None',
        merge_pages=2
    )
    
    result = doc_processor.process_documents(setup_fixture['docs'])
    
    page_ids = [DocumentProcessor.stable_hash_meta(page.metadata) for page in result.pages]
    chunk_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in result.chunks]

    assert result.rag_type == setup_fixture['rag_type']['Standard']
    assert result.pages is not None
    assert len(page_ids) == len(set(page_ids))
    assert result.chunks is not None
    assert chunk_ids == page_ids
    assert result.splitters is None
def test_process_documents_nochunk(setup_fixture):
    '''Test document processing with no chunking.'''
    logger = setup_fixture['logger']
    logger.info("Starting process_documents_nochunk test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        rag_type=setup_fixture['rag_type']['Standard'],
        chunk_method='None',
        chunk_size=setup_fixture['chunk_size'],
        chunk_overlap=setup_fixture['chunk_overlap']
    )
    
    result = doc_processor.process_documents(setup_fixture['docs'])
    
    page_ids = [DocumentProcessor.stable_hash_meta(page.metadata) for page in result.pages]
    chunk_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in result.chunks]

    assert result.rag_type == setup_fixture['rag_type']['Standard']
    assert result.pages is not None
    assert len(page_ids) == len(set(page_ids))
    assert result.chunks is result.pages
    assert len(chunk_ids) == len(set(chunk_ids))
    assert result.splitters is None
def test_process_documents_parent_child(setup_fixture):
    '''Test document processing with parent-child RAG.'''
    logger = setup_fixture['logger']
    logger.info("Starting process_documents_parent_child test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        rag_type=setup_fixture['rag_type']['Parent-Child'],
        chunk_method=setup_fixture['chunk_method'],
        chunk_size=setup_fixture['chunk_size'],
        chunk_overlap=setup_fixture['chunk_overlap']
    )
    
    result = doc_processor.process_documents(setup_fixture['docs'])
    
    page_ids = [DocumentProcessor.stable_hash_meta(page.metadata) for page in result.pages['parent_chunks']]
    chunk_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in result.chunks]

    assert result.rag_type == setup_fixture['rag_type']['Parent-Child']
    assert result.pages['doc_ids'] is not None
    assert result.pages['parent_chunks'] is not None
    assert len(page_ids) == len(set(page_ids))
    assert result.chunks is not None
    assert len(chunk_ids) == len(set(chunk_ids))
    assert result.splitters['parent_splitter'] is not None
    assert result.splitters['child_splitter'] is not None
def test_process_documents_summary(setup_fixture):
    '''Test document processing with summary RAG.'''
    logger = setup_fixture['logger']
    logger.info("Starting process_documents_summary test")
    
    # Test case 1: Without LLM service (should raise error)
    with pytest.raises(ValueError, match="LLM service is required for Summary RAG type"):
        doc_processor = DocumentProcessor(
            embedding_service=setup_fixture['mock_embedding_service'],
            rag_type=setup_fixture['rag_type']['Summary'],
            chunk_method=setup_fixture['chunk_method'],
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            llm_service=None  # No LLM service provided
        )
        doc_processor.process_documents(setup_fixture['docs'])
    
    # Test case 2: With LLM service (should succeed)
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        rag_type=setup_fixture['rag_type']['Summary'],
        chunk_method=setup_fixture['chunk_method'],
        chunk_size=setup_fixture['chunk_size'],
        chunk_overlap=setup_fixture['chunk_overlap'],
        llm_service=setup_fixture['mock_llm_service']
    )
    
    result = doc_processor.process_documents(setup_fixture['docs'])
    
    page_ids = [DocumentProcessor.stable_hash_meta(page.metadata) for page in result.pages['docs']]
    summary_ids = [DocumentProcessor.stable_hash_meta(summary.metadata) for summary in result.summaries]
    
    assert result.rag_type == setup_fixture['rag_type']['Summary']
    assert result.pages['doc_ids'] is not None
    assert result.pages['docs'] is not None
    assert len(page_ids) == len(set(page_ids))
    assert result.summaries is not None
    assert len(summary_ids) == len(set(summary_ids))
    assert result.llm_service == setup_fixture['mock_llm_service']
@pytest.mark.parametrize('test_index', [
    {
        'index_type': 'Pinecone',
        'embedding_family': 'OpenAI',
        'embedding_name': 'text-embedding-3-small',
        'expected_class': PineconeVectorStore
    },
    {
        'index_type': 'ChromaDB',
        'embedding_family': 'OpenAI',
        'embedding_name': 'text-embedding-ada-002',
        'expected_class': Chroma
    },
    {
        'index_type': 'RAGatouille',
        'embedding_family': 'RAGatouille',
        'embedding_name': 'colbert-ir/colbertv2.0',
        'expected_class': RAGPretrainedModel
    }
])
def test_initialize_database(monkeypatch, test_index):
    '''Test the initialization of different types of databases.'''
    logger = logging.getLogger(__name__)
    logger.info(f"Starting initialize_database test with {test_index['index_type']}")
    
    index_name = 'test-index'
    rag_type = 'Standard'

    # Create services
    embedding_service = EmbeddingService(
        model_name=test_index['embedding_name'],
        model_type=test_index['embedding_family']
    )
    
    db_service = DatabaseService(
        db_type=test_index['index_type'],
        index_name=index_name,
        rag_type=rag_type,
        embedding_service=embedding_service,
        doc_type='document'
    )

    # Clean up any existing database first
    try:
        db_service.delete_index()
    except:
        pass  # Ignore errors if database doesn't exist

    # Test with environment variable local_db_path
    try:
        db_service.initialize_database(
            clear=True
        )

        assert isinstance(db_service.vectorstore, test_index['expected_class'])
        
        # Cleanup
        db_service.delete_index()

    except Exception as e:
        # If there is an error, be sure to delete the database
        try:
            db_service.delete_index()
        except:
            pass
        raise e

    # Test with local_db_path set manually, show it doesn't work if not set
    monkeypatch.delenv('LOCAL_DB_PATH', raising=False)
    with pytest.raises(ValueError,match="LOCAL_DB_PATH environment variable must be set"):
        db_service = DatabaseService(
            db_type=test_index['index_type'],
            index_name=index_name,
            rag_type=rag_type,
            embedding_service=embedding_service,
            doc_type='document'
        )
        db_service.initialize_database(
            namespace=db_service.namespace,
            clear=True
        )
@pytest.mark.parametrize('test_index', [
    # Combined Pinecone and ChromaDB tests
    {
        'db_types': ['Pinecone', 'ChromaDB'],
        'embedding_family': 'OpenAI',
        'embedding_name': 'text-embedding-3-small',
        'expected_classes': {
            'Pinecone': PineconeVectorStore,
            'ChromaDB': Chroma
        },
        'rag_types': ['Standard', 'Parent-Child', 'Summary']
    },
    # RAGatouille test (Standard only)
    {
        'db_types': ['RAGatouille'],
        'embedding_family': 'RAGatouille',
        'embedding_name': 'colbert-ir/colbertv2.0',
        'expected_classes': {
            'RAGatouille': RAGPretrainedModel
        },
        'rag_types': ['Standard']
    }
])
def test_delete_database(setup_fixture, test_index):
    '''Test deleting both existing and non-existing databases for different RAG types.'''
    logger = setup_fixture['logger']
    logger.info(f"Starting delete_database test.")
    
    index_name = 'test-delete-index'

    # Create embedding service
    embedding_service = EmbeddingService(
        model_name=test_index['embedding_name'],
        model_type=test_index['embedding_family']
    )
    
    # Loop through each database type and RAG type combination
    for db_type in test_index['db_types']:
        for rag_type in test_index['rag_types']:
            logger.info(f"\nTesting {db_type} with {rag_type} RAG type")
            
            db_service = DatabaseService(
                db_type=db_type,
                index_name=index_name,
                rag_type=rag_type,
                embedding_service=embedding_service,
                doc_type='document'
            )

            # Clean up any existing test indexes first
            logger.info(f"Deleting existing indexes before test cases: {db_service.index_name}")
            try:
                db_service.delete_index()
            except Exception as e:
                logger.info(f"Info: Cleanup of existing index failed (this is expected if index didn't exist): {str(e)}")

            # Test Case 1: Delete non-existent database
            logger.info(f"Deleting non-existent database: {db_service.index_name}")
            try:
                db_service.delete_index()
            except Exception as e:
                assert "does not exist" in str(e).lower() or "not found" in str(e).lower()

            # Test Case 2: Create and delete database with specific RAG type
            logger.info(f"Creating and deleting {rag_type} database: {db_service.index_name}")
            try:
                db_service.initialize_database(
                    namespace=db_service.namespace,
                    clear=True
                )
                assert isinstance(db_service.vectorstore, test_index['expected_classes'][db_type])
                
                # Delete the database
                db_service.delete_index()
                
                # Verify deletion by checking if database exists
                if db_type == 'Pinecone':
                    pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
                    assert db_service.index_name not in pc.list_indexes()
                    
                    # For Parent-Child and Summary, verify additional indexes are deleted
                    if rag_type in ['Parent-Child', 'Summary']:
                        lfs_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'local_file_store', index_name)
                        assert not os.path.exists(lfs_path)
                        
                elif db_type == 'ChromaDB':
                    db_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], db_service.index_name)
                    assert not os.path.exists(db_path)
                    
                    # For Parent-Child and Summary, verify additional storage is deleted
                    if rag_type in ['Parent-Child', 'Summary']:
                        lfs_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'local_file_store', index_name)
                        assert not os.path.exists(lfs_path)
                        
                elif db_type == 'RAGatouille':
                    db_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'ragatouille', db_service.index_name)
                    assert not os.path.exists(db_path)

                # Verify index is not in available indexes
                available_indexes, _ = get_available_indexes(
                    db_type,
                    test_index['embedding_name'],
                    rag_type
                )
                assert db_service.index_name not in available_indexes, \
                    f"Deleted index {db_service.index_name} still appears in available indexes"

            except Exception as e:
                # If test fails, ensure cleanup
                try:
                    db_service.delete_index()
                except:
                    pass
                raise e
def test_rag_type_mismatch(setup_fixture):
    """Test handling of mismatched RAG types between DatabaseService and ChunkingResult."""
    logger = setup_fixture['logger']
    logger.info("Starting rag_type_mismatch test")

    try:
        # Test case 1: Attempt to index data with mismatched RAG types
        db_service = DatabaseService(
            db_type='ChromaDB',
            index_name='test-rag-mismatch',
            rag_type='Standard',
            embedding_service=setup_fixture['mock_embedding_service'],
            doc_type='document'
        )
        db_service.initialize_database(clear=True)
        doc_processor = DocumentProcessor(
            embedding_service=setup_fixture['mock_embedding_service'],
            rag_type='Parent-Child',  # Intentionally different from db_service
            chunk_method=setup_fixture['chunk_method'],
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap']
        )
        # Process documents
        chunking_result = doc_processor.process_documents(setup_fixture['docs'])
        with pytest.raises(ValueError, match="RAG type mismatch"):
            db_service.index_data(
                data=chunking_result,
                batch_size=setup_fixture['batch_size']
            )
        db_service.delete_index()

        # Test case 2: Reverse mismatch (Parent-Child DB with Standard chunks)
        db_service = DatabaseService(
            db_type='ChromaDB',
            index_name='test-rag-mismatch',
            rag_type='Parent-Child',
            embedding_service=setup_fixture['mock_embedding_service'],
            doc_type='document'
        )
        db_service.initialize_database(clear=True)
        doc_processor = DocumentProcessor(
            embedding_service=setup_fixture['mock_embedding_service'],
            rag_type='Standard',  # Intentionally different from db_service
            chunk_method=setup_fixture['chunk_method'],
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap']
        )
        chunking_result = doc_processor.process_documents(setup_fixture['docs'])
        with pytest.raises(ValueError, match="RAG type mismatch"):
            db_service.index_data(
                data=chunking_result,
                batch_size=setup_fixture['batch_size']
            )

    finally:
        # Cleanup
        db_service.delete_index()
@pytest.mark.parametrize('test_index', [
    {
        'index_type': 'Pinecone',
        'embedding_model': 'text-embedding-3-small',
        'embedding_family': 'OpenAI',
        'rag_types': ['Standard', 'Parent-Child', 'Summary']
    },
    {
        'index_type': 'ChromaDB',
        'embedding_model': 'text-embedding-3-small',
        'embedding_family': 'OpenAI',
        'rag_types': ['Standard', 'Parent-Child', 'Summary']
    },
    {
        'index_type': 'RAGatouille',
        'embedding_model': 'colbert-ir/colbertv2.0',
        'embedding_family': 'RAGatouille',
        'rag_types': ['Standard']  # RAGatouille only supports Standard
    }
])
def test_get_available_indexes(setup_fixture, test_index):
    """Test retrieving available indexes for different database and RAG configurations."""
    logger = setup_fixture['logger']
    logger.info(f"Starting get_available_indexes test with {test_index['index_type']}")
    
    # Create services
    embedding_service = EmbeddingService(
        model_name=test_index['embedding_model'],
        model_type=test_index['embedding_family']
    )
    
    llm_service = LLMService(
        model_name='gpt-4o-mini',
        model_type='OpenAI'
    )

    # Create and index test documents for each RAG type
    test_indexes = []
    for rag_type in test_index['rag_types']:
        index_name_initialized = f"test-{test_index['index_type'].lower()}"
        logger.info(f"Creating test index: {index_name_initialized} for {rag_type} RAG type")
        try:
            # Initialize database service
            db_service = DatabaseService(
                db_type=test_index['index_type'],
                index_name=index_name_initialized,
                rag_type=rag_type,
                embedding_service=embedding_service,
                doc_type='document'
            )
            db_service.initialize_database(clear=True)

            # Initialize document processor
            doc_processor = DocumentProcessor(
                embedding_service=embedding_service,
                rag_type=rag_type,
                chunk_method=setup_fixture['chunk_method'],
                chunk_size=setup_fixture['chunk_size'],
                chunk_overlap=setup_fixture['chunk_overlap'],
                llm_service=llm_service if rag_type == 'Summary' else None
            )

            # Process and index documents
            chunking_result = doc_processor.process_documents(setup_fixture['docs'])
            db_service.index_data(
                data=chunking_result,
                batch_size=setup_fixture['batch_size']
            )
            test_indexes.append(db_service.index_name)

        except Exception as e:
            print(f"Error creating index {index_name_initialized}: {str(e)}")
            db_service.delete_index()
            raise e

    # try:
    # Test getting available indexes for each RAG type
    for i, rag_type in enumerate(test_index['rag_types']):
        logger.info(f"Testing for getting available indexes for {rag_type} RAG type")
        try:
            index_name_checked = f"test-{test_index['index_type'].lower()}"

            # FIXME, the get_available_indexes function is not returning anything
            available_indexes, index_metadatas = get_available_indexes(
                test_index['index_type'],
                test_index['embedding_model'],
                rag_type
            )
                            
            # Construct expected index name based on RAG type
            if rag_type == 'Parent-Child':
                expected_index = f"{index_name_checked}-parent-child"
            elif rag_type == 'Summary':
                expected_index = f"{index_name_checked}-summary"
            else:  # Standard
                expected_index = index_name_checked
            
            logger.info(f"Available {rag_type} indexes: {available_indexes}. Checking for {expected_index}")
            
            # Verify expected index exists in results
            assert expected_index in available_indexes, \
                f"Expected index {expected_index} not found in available indexes"
            
            # Verify metadata matches
            if test_index['index_type'] != 'RAGatouille':
                for index_metadata in index_metadatas:
                    assert index_metadata['embedding_model'] == test_index['embedding_model']

        except Exception as e:
            db_service = DatabaseService(
                db_type=test_index['index_type'],
                index_name=test_indexes[i],
                rag_type=rag_type,
                embedding_service=embedding_service,
                doc_type='document'
            )
            db_service.delete_index()
            logger.error(f"Cleanup: Deleted test index: {test_indexes[i]}")
            raise e
        
        finally:
            # Clean up the current test index
            try:
                db_service = DatabaseService(
                    db_type=test_index['index_type'],
                    index_name=test_indexes[i],
                    rag_type=rag_type,
                    embedding_service=embedding_service,
                    doc_type='document'
                )
                db_service.delete_index()
                logger.error(f"Deleted test index: {test_indexes[i]}")
            except Exception as e:
                logger.error(f"Error deleting index {test_indexes[i]}: {str(e)}")
def test_database_setup_and_query(test_input, setup_fixture):
    '''Tests the entire process of initializing a database, upserting documents, and deleting a database.'''
    logger = setup_fixture['logger']
    test, print_str = parse_test_case(setup_fixture, test_input)
    logger.info(f"Starting database_setup_and_query test: {print_str}")

    from aerospace_chatbot.services.database import DatabaseService
    from aerospace_chatbot.processing import DocumentProcessor

    index_name = 'test' + str(test['id'])
    logger.info(f'Starting test: {print_str}')

    # Get services
    embedding_service = EmbeddingService(
        model_name=test['embedding_name'],
        model_type=test['embedding_family']
    )
    llm_service = LLMService(
        model_name=test['llm'],
        model_type=test['llm_family']
    )

    db_service = DatabaseService(
        db_type=test['index_type'],
        index_name=index_name,
        rag_type=test['rag_type'],
        embedding_service=embedding_service,
        doc_type='document'
    )

    try:
        # Initialize the document processor with services
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            rag_type=test['rag_type'],
            chunk_method=setup_fixture['chunk_method'],
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            llm_service=llm_service
        )
        # Process and index documents
        db_service.initialize_database(
            clear=True
        )
        # Verify the vectorstore type
        if db_service.db_type == 'ChromaDB':
            assert isinstance(db_service.vectorstore, Chroma)
        elif db_service.db_type == 'Pinecone':
            assert isinstance(db_service.vectorstore, PineconeVectorStore)
        elif db_service.db_type == 'RAGatouille':
            assert isinstance(db_service.vectorstore, RAGPretrainedModel)
        logger.info('Vectorstore created.')

        # Process and index documents
        chunking_result = doc_processor.process_documents(setup_fixture['docs'])
        db_service.index_data(
            data=chunking_result,
            batch_size=setup_fixture['batch_size']
        )

        # Initialize QA model
        qa_model = QAModel(
            db_service=db_service,
            llm_service=llm_service
        )
        logger.info('QA model object created.')
        assert qa_model is not None

        # Run a query and verify results
        qa_model.query(setup_fixture['test_prompt'])
        assert qa_model.ai_response is not None
        assert qa_model.sources is not None
        assert qa_model.memory is not None

        # Generate alternative questions
        alternate_question = qa_model.generate_alternative_questions(setup_fixture['test_prompt'])
        assert alternate_question is not None
        logger.info('Query and alternative question successful!')

        # Delete the indexes
        db_service.delete_index()
        qa_model.query_db_service.delete_index()  # Delete the query database index
        
        if doc_processor.rag_type in ['Parent-Child', 'Summary']:
            lfs_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'local_file_store', index_name)
            assert not os.path.exists(lfs_path)  # Check that the local file store was deleted
        logger.info('Databases deleted.')

    except Exception as e:  # If there is an error, be sure to delete the database
        db_service.delete_index()
        raise e

### Frontend tests
def test_sidebar_manager(setup_fixture):
    """Test the SidebarManager class functionality."""
    logger = setup_fixture['logger']
    logger.info("Starting sidebar_manager test")
    
    # Setup
    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file = os.path.join(home_dir, 'config', 'config_admin.json')

    # Test initialization
    sidebar_manager = SidebarManager(config_file)
    assert sidebar_manager._config is not None
    assert 'databases' in sidebar_manager._config
    assert 'embeddings' in sidebar_manager._config
    assert 'llms' in sidebar_manager._config
    assert 'rag_types' in sidebar_manager._config

    # Test single case since render_sidebar now renders everything
    sb_out = sidebar_manager.render_sidebar()
    
    # Verify all outputs are present since everything is rendered
    # Core dependencies
    assert 'index_type' in sb_out
    assert 'embedding_name' in sb_out
    assert 'rag_type' in sb_out
    
    # Embeddings outputs
    assert 'embedding_model' in sb_out
    assert 'embedding_name' in sb_out
    
    # RAG type outputs  
    assert 'rag_type' in sb_out
    
    # LLM outputs
    assert 'llm_source' in sb_out
    assert 'llm_model' in sb_out
    
    # Model options outputs
    assert 'model_options' in sb_out
    assert 'temperature' in sb_out['model_options']
    assert 'output_level' in sb_out['model_options']
    assert 'k' in sb_out['model_options']
def test_sidebar_manager_invalid_config(setup_fixture):
    """Test SidebarManager initialization with invalid config file path"""
    logger = setup_fixture['logger']
    logger.info("Starting sidebar_manager_invalid_config test")
    
    with pytest.raises(ConfigurationError):
        SidebarManager('nonexistent_config.json')
def test_sidebar_manager_malformed_config(setup_fixture):
    """Test SidebarManager with malformed config file"""
    logger = setup_fixture['logger']
    logger.info("Starting sidebar_manager_malformed_config test")
    
    # Create a temporary malformed config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tf:
        tf.write('{"invalid": "json"')
        tf.flush()
        with pytest.raises(ConfigurationError):
            SidebarManager(tf.name)
def test_sidebar_manager_missing_required_sections(setup_fixture):
    """Test SidebarManager with config missing required sections"""
    logger = setup_fixture['logger']
    logger.info("Starting sidebar_manager_missing_required_sections test")
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tf:
        # Create config missing 'databases' section
        json.dump({'embeddings': {}, 'llms': {}, 'rag_types': {}}, tf)
        tf.flush()
        with pytest.raises(ConfigurationError):
            SidebarManager(tf.name)
def test_set_secrets_with_valid_input(setup_fixture):
    '''Test case for set_secrets function with valid input.'''
    logger = setup_fixture['logger']
    logger.info("Starting set_secrets_with_valid_input test")
    
    test_secrets = {
        'OPENAI_API_KEY': 'openai_key',
        'VOYAGE_API_KEY': 'voyage_key',
        'PINECONE_API_KEY': 'pinecone_key',
        'HUGGINGFACEHUB_API_KEY': 'huggingface_key',
        'ANTHROPIC_API_KEY': 'anthropic_key'
    }
    
    # Set secrets and verify return
    result = set_secrets(test_secrets)
    assert result == test_secrets
    
    # Verify environment variables were set
    for key, value in test_secrets.items():
        assert os.environ[key] == value
@pytest.mark.parametrize('test_index', [
    {
        'db_type': 'ChromaDB',
        'embedding_family': 'OpenAI',
        'embedding_name': 'text-embedding-3-small'
    },
    {
        'db_type': 'Pinecone',
        'embedding_family': 'OpenAI',
        'embedding_name': 'text-embedding-3-small'
    }
])
def test_get_docs_questions_df(setup_fixture, test_index):
    """Test function for the get_docs_questions_df() method."""
    logger = setup_fixture['logger']
    logger.info(f"Starting get_docs_questions_df test with {test_index['db_type']}")
    
    index_name = 'test-visualization-df_export'
    rag_type='Standard'

    # Initialize services
    embedding_service = EmbeddingService(
        model_name=test_index['embedding_name'],
        model_type=test_index['embedding_family']
    )
    llm_service = LLMService(
        model_name='gpt-4o-mini',
        model_type='OpenAI'
    )
    db_service = DatabaseService(
        db_type=test_index['db_type'],
        index_name=index_name,
        rag_type=rag_type,
        embedding_service=embedding_service,
        doc_type='document'
    )
    db_service.initialize_database(clear=True)

    try:
        # Initialize the document processor with services
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            rag_type=rag_type,
            chunk_method=setup_fixture['chunk_method'],
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            llm_service=llm_service
        )

        # Process and index documents
        chunking_result = doc_processor.process_documents(setup_fixture['docs'])
        db_service.index_data(
            data=chunking_result,
            batch_size=setup_fixture['batch_size']
        )

        # Initialize QA model
        n_retrievals = 4
        qa_model = QAModel(
            db_service=db_service,
            llm_service=llm_service,
            k=n_retrievals
        )

        # Run a query to create the query database
        qa_model.query(setup_fixture['test_prompt'])

        # Get combined dataframe
        df = get_docs_questions_df(
            db_service=db_service,  # Main document database service
            query_db_service=qa_model.query_db_service  # Query database service from QA model
        )

        # Assert the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in [
            "id", "source", "page", "document", "embedding", "type",
            "first_source", "used_by_questions", "used_by_num_questions",
            "used_by_question_first"
        ])

        # Verify there is at least one document and one question
        assert len(df[df['type'] == 'doc']) > 0, "No documents found in DataFrame"
        assert len(df[df['type'] == 'question']) > 0, "No questions found in DataFrame" # Indicates the question was not recorded properly
        
        # Check that exactly n_retrievals sources were used (have non-zero values in used_by_num_questions)
        nonzero_count = len(df[df['used_by_num_questions'] > 0])
        assert nonzero_count == n_retrievals, f"Expected {n_retrievals} sources to have non-zero values in used_by_num_questions, but got {nonzero_count}"

        # Cleanup
        db_service.delete_index()
        qa_model.query_db_service.delete_index()
        print(f'Database deleted: {test_index["db_type"]}')

    except Exception as e:  # If there is an error, be sure to delete the database
        try:
            db_service.delete_index()
            qa_model.query_db_service.delete_index()
        except:
            pass
        raise e
@pytest.mark.parametrize('test_index', [
    {
        'db_type': 'ChromaDB',
        'embedding_family': 'OpenAI',
        'embedding_name': 'text-embedding-3-small'
    },
    {
        'db_type': 'Pinecone',
        'embedding_family': 'OpenAI',
        'embedding_name': 'text-embedding-3-small'
    }
])
def test_add_clusters(setup_fixture, test_index):
    """Test function for the add_clusters function."""
    logger = setup_fixture['logger']
    logger.info(f"Starting add_clusters test with {test_index['db_type']}")
    
    index_name = 'test-visualization-add-clusters'

    # Initialize services
    embedding_service = EmbeddingService(
        model_name=test_index['embedding_name'],
        model_type=test_index['embedding_family']
    )
    llm_service = LLMService(
        model_name='gpt-4o-mini',
        model_type='OpenAI'
    )
    db_service = DatabaseService(
        db_type=test_index['db_type'],
        index_name=index_name,
        rag_type='Standard',
        embedding_service=embedding_service,
        doc_type='document'
    )

    try:
        # Initialize the document processor with services
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            rag_type='Standard',
            chunk_method=setup_fixture['chunk_method'],
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            llm_service=llm_service
        )

        # Process and index documents
        db_service.initialize_database(
            clear=True
        )
        chunking_result = doc_processor.process_documents(setup_fixture['docs'])
        db_service.index_data(
            data=chunking_result,
            batch_size=setup_fixture['batch_size']
        )

        # Initialize QA model
        qa_model = QAModel(
            db_service=db_service,
            llm_service=llm_service
        )

        # Run a query to create the query database
        qa_model.query(setup_fixture['test_prompt'])

        # Get combined dataframe
        df = get_docs_questions_df(
            db_service=db_service,  # Main document database service
            query_db_service=qa_model.query_db_service  # Query database service from QA model
        )

        # Test clustering without labels
        n_clusters = 2
        df_with_clusters = add_clusters(df, n_clusters)
        assert len(df_with_clusters["cluster"].unique()) == n_clusters
        for cluster in df_with_clusters["cluster"].unique():
            assert len(df_with_clusters[df_with_clusters["cluster"] == cluster]) >= 1

        # Test clustering with labels
        df_with_clusters = add_clusters(df, n_clusters, llm_service, 2)
        assert len(df_with_clusters["cluster"].unique()) == n_clusters
        assert "cluster_label" in df_with_clusters.columns
        assert df_with_clusters["cluster_label"].notnull().all()
        assert df_with_clusters["cluster_label"].apply(lambda x: isinstance(x, str)).all()
        for cluster in df_with_clusters["cluster"].unique():
            assert len(df_with_clusters[df_with_clusters["cluster"] == cluster]) > 0

        # Cleanup
        db_service.delete_index()
        qa_model.query_db_service.delete_index()
        print(f'Database deleted: {test_index["db_type"]}')

    except Exception as e:  # If there is an error, be sure to delete the database
        try:
            db_service.delete_index()
            qa_model.query_db_service.delete_index()
        except:
            pass
        raise e