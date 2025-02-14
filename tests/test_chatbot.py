import os, sys, json
import itertools
import pytest
import pandas as pd
import logging
from dotenv import load_dotenv, find_dotenv

from pinecone import Pinecone as pinecone_client
from ragatouille import RAGPretrainedModel

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser

from aerospace_chatbot.core import (
    ConfigurationError,
    set_secrets
)
from aerospace_chatbot.processing import (
    DocumentProcessor, 
    QAModel
)
from aerospace_chatbot.services import (
    DatabaseService, 
    get_docs_df, 
    add_clusters,
    get_available_indexes,
    DatabaseService,
    EmbeddingService, 
    RerankService,
    LLMService,
    InLineCitationsResponse,
    CHATBOT_SYSTEM_PROMPT,
    QA_PROMPT,
    SUMMARIZE_TEXT,
    GENERATE_SIMILAR_QUESTIONS_W_CONTEXT,
    CLUSTER_LABEL,
    DEFAULT_DOCUMENT_PROMPT,
)
from aerospace_chatbot.ui import (
    SidebarManager
)
from aerospace_chatbot.ui.utils import process_uploads

# Import local variables
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src/aerospace_chatbot'))

# Low priority updates:
# TODO add tests to check conversation history functionality
# TODO add apptest from streamlit
# TODO add partitioned docs for faster testing to avoid doc parsing and indexing each time.

def permute_tests(test_data):
    """
    Generate permutations of test cases.
    """
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
    """
    Use pytest_generate_tests to dynamically generate tests.
    Tests generates tests from a static file (test_cases.json). See test_cases.json for more details.
    """
    if 'test_input' in metafunc.fixturenames:
        tests = read_test_cases(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_cases.json'))
        metafunc.parametrize('test_input', tests)
def parse_test_case(setup, test_case):
    """
    Parse test case to be used in the test functions.
    """
    parsed_test = {
        'id': test_case['id'],
        'index_type': setup['index_type'][test_case['index_type']],
        'embedding_service': test_case['embedding_service'],
        'embedding_model': test_case['embedding_model'],
        'rerank_service': test_case.get('rerank_service', None),
        'rerank_model': test_case.get('rerank_model', None),
        'llm_service': test_case['llm_service'],
        'llm_model': test_case['llm_model']
    }
    print_str = ', '.join(f'{key}: {value}' for key, value in test_case.items())

    return parsed_test, print_str

@pytest.fixture(autouse=True)
def setup_fixture():
    """
    Sets up the fixture for testing the backend.
    """
    # Store original environment variables
    original_env = {
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY'),
        'VOYAGE_API_KEY': os.environ.get('VOYAGE_API_KEY'),
        'HUGGINGFACEHUB_API_KEY': os.environ.get('HUGGINGFACEHUB_API_KEY'),
        'PINECONE_API_KEY': os.environ.get('PINECONE_API_KEY'),
        'UNSTRUCTURED_API_KEY': os.environ.get('UNSTRUCTURED_API_KEY'),
        'UNSTRUCTURED_API_URL': os.environ.get('UNSTRUCTURED_API_URL'),
        'LOG_LEVEL': os.environ.get('LOG_LEVEL'),
        'LOG_FILE': os.environ.get('LOG_FILE'),
        'LOCAL_DB_PATH': os.environ.get('LOCAL_DB_PATH')
    }

    # Override environment variables first
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['LOG_FILE'] = 'logs/test_chatbot.log'

    # Override admin vs. test
    os.environ['AEROSPACE_CHATBOT_CONFIG'] = 'admin'
    
    # Setup logging with overridden values
    from aerospace_chatbot.core.config import setup_logging
    logger = setup_logging()
    
    # Now load .env file (won't override existing environment variables)
    load_dotenv(find_dotenv(), override=False)
    
    # Set mock API keys
    mock_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY'),
        'HUGGINGFACEHUB_API_KEY': os.getenv('HUGGINGFACEHUB_API_KEY'),
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
        'UNSTRUCTURED_API_KEY': os.getenv('UNSTRUCTURED_API_KEY'),
        'UNSTRUCTURED_API_URL': os.getenv('UNSTRUCTURED_API_URL')
    }
    
    # Set environment variables
    for key, value in mock_keys.items():
        os.environ[key] = value

    LOCAL_DB_PATH = os.path.abspath(os.path.dirname(__file__))
    os.environ['LOCAL_DB_PATH'] = LOCAL_DB_PATH
    
    # Fixed inputs
    docs = ['test1.pdf', 'test2.pdf']
    for i in range(len(docs)):
        docs[i] = os.path.join(os.path.abspath(os.path.dirname(__file__)), docs[i])

    chunk_size = 400
    chunk_overlap = 0
    batch_size = 50
    test_prompt = 'What are some nuances associated with the analysis and design of hinged booms?'   # Info on test2.pdf

    # Variable inputs
    index_type = {index: index for index in ['Pinecone', 'RAGatouille']}
    
    mock_embedding_service = EmbeddingService(
        model_service='OpenAI',
        model='text-embedding-3-small'
    )

    mock_rerank_service = RerankService(
        model_service='Cohere',
        model='rerank-v3.5'
    )

    mock_llm_service = LLMService(
        model_service='OpenAI',
        model='gpt-4o-mini',
        temperature=0,
        max_tokens=5000
    )

    setup = {
        'logger': logger,
        **mock_keys,
        'LOCAL_DB_PATH': LOCAL_DB_PATH,
        'docs': docs,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'batch_size': batch_size,
        'test_prompt': test_prompt,
        'index_type': index_type,
        # Add mock services
        'mock_embedding_service': mock_embedding_service,
        'mock_rerank_service': mock_rerank_service,
        'mock_llm_service': mock_llm_service
    }

    yield setup

    # Restore original environment variables after test
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

def test_validate_index(setup_fixture):
    """
    Test edge cases for validate_index function.
    """    
    logger = setup_fixture['logger']
    logger.info("Starting validate_index test")

    # Test case 1: Empty index name
    db_type='Pinecone'
    db_service = DatabaseService(
        db_type=db_type,
        index_name='',
        embedding_service=setup_fixture['mock_embedding_service'],
        rerank_service=setup_fixture['mock_rerank_service']
    )
    
    with pytest.raises(ValueError, match="Index name cannot be empty"):
        db_service.index_name = ""
        db_service._validate_index()

    # Test case: Whitespace-only index name
    with pytest.raises(ValueError, match="Index name cannot be empty"):
        db_service.index_name = "   "
        db_service._validate_index()

    # Test case: consecutive periods
    with pytest.raises(ValueError, match="cannot contain underscores"):
        db_service.index_name = "test_index"
        db_service._validate_index()

    # Test case: name too long
    with pytest.raises(ValueError, match="must be less than 45 characters"):
        db_service.index_name = "a" * 46
        db_service._validate_index()

def test_load_documents(setup_fixture):
    """
    Test document loading with various input types and edge cases.
    """
    logger = setup_fixture['logger']
    logger.info("Starting load_documents test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service']
    )
    
    # Test loading local PDFs
    local_docs, gcs_docs = doc_processor._load_documents(setup_fixture['docs'])
    assert len(local_docs) == len(setup_fixture['docs'])
    assert all(os.path.exists(doc) for doc in local_docs)
    assert len(gcs_docs) == 0  # No GCS docs in test setup
    
    # Test invalid PDF
    invalid_path = os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'invalid.pdf')
    with open(invalid_path, 'w') as f:
        f.write('Not a PDF')
    
    with pytest.raises(ValueError, match="Invalid PDF documents detected"):
        doc_processor._load_documents([invalid_path])
    
    # Test non-PDF file
    with pytest.raises(ValueError, match="Invalid PDF documents detected"):
        doc_processor._load_documents(['test.txt'])
    
    # Cleanup
    try:
        os.remove(invalid_path)
    except:
        pass
def test_load_partitioned_documents(setup_fixture):
    """
    Test loading partitioned documents with various scenarios.
    """
    logger = setup_fixture['logger']
    logger.info("Starting load_partitioned_documents test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service']
    )
    
    # Test Case 1: Loading from existing partitioned files in test_processed_docs
    partition_dir = os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
    assert os.path.exists(partition_dir), "test_processed_docs directory should exist"
    
    partitioned_docs = doc_processor.load_partitioned_documents(
        setup_fixture['docs'],
        partition_dir=partition_dir
    )
    # Verify the files exist and have correct format
    assert len(partitioned_docs) > 0, "Should find partitioned documents"
    for doc in partitioned_docs:
        assert os.path.exists(doc), f"Partitioned file {doc} should exist"
        assert doc.endswith('-partitioned.json'), f"File {doc} should end with -partitioned.json"
        # Verify the JSON is valid and has expected structure
        with open(doc, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list), "Partitioned file should contain a list"
            assert len(data) > 0, "Partitioned file should not be empty"
            # Check first element has expected keys
            assert all(key in data[0] for key in ['text', 'type']), "Partitioned data should have required fields"
    
    # Test Case 2: Loading with non-existent partition directory
    non_existent_dir = os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'non_existent_dir')
    with pytest.raises(FileNotFoundError):
        doc_processor.load_partitioned_documents(
            setup_fixture['docs'],
            partition_dir=non_existent_dir
        )
    
    # Test Case 3: Loading with empty partition directory
    empty_dir = os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'empty_dir')
    os.makedirs(empty_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        doc_processor.load_partitioned_documents(
            setup_fixture['docs'],
            partition_dir=empty_dir
        )
    
    # Test Case 4: Loading with invalid JSON in partitioned file
    invalid_json_path = os.path.join(partition_dir, 'invalid-partitioned.json')
    with open(invalid_json_path, 'w') as f:
        f.write('{"invalid": json}')  # Intentionally malformed JSON
    
    with pytest.raises(FileNotFoundError):
        doc_processor.load_partitioned_documents(
            [invalid_json_path],
            partition_dir=partition_dir
        )
    
    # Cleanup only the files we created for testing
    try:
        os.remove(invalid_json_path)
        os.rmdir(empty_dir)
    except:
        pass

def test_partition_methods(setup_fixture):
    """
    Test both API and local partitioning methods.
    """
    logger = setup_fixture['logger']
    logger.info("Starting partition_methods test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        work_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'document_processing') # Put partitioned files in same directory as original files
    )
    
    # Test API partitioning
    api_partitioned = doc_processor.load_and_partition_documents(
        setup_fixture['docs'],
        partition_by_api=True
    )
    assert len(api_partitioned) == len(setup_fixture['docs'])
    assert all(os.path.exists(doc) for doc in api_partitioned)
    assert all(doc.endswith('-partitioned.json') for doc in api_partitioned)
    
    # Test local partitioning
    local_partitioned = doc_processor.load_and_partition_documents(
        setup_fixture['docs'],
        partition_by_api=False
    )
    assert len(local_partitioned) == len(setup_fixture['docs'])
    assert all(os.path.exists(doc) for doc in local_partitioned)
    assert all(doc.endswith('-partitioned.json') for doc in local_partitioned)
    
    # Compare results (should have similar structure)
    with open(api_partitioned[0], 'r') as f:
        api_data = json.load(f)
    with open(local_partitioned[0], 'r') as f:
        local_data = json.load(f)
    
    assert len(api_data) > 0
    assert len(local_data) > 0
    # Check that both methods produce similar structure
    assert set(api_data[0].keys()) == set(local_data[0].keys())

def test_chunking_result(setup_fixture):
    """
    Test ChunkingResult class functionality.
    """
    logger = setup_fixture['logger']
    logger.info("Starting chunking_result test")
    
    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        work_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'document_processing') # Put partitioned files in same directory as original files
    )
    
    # Get chunks
    partitioned_docs = doc_processor.load_partitioned_documents(
        setup_fixture['docs'],
        partition_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
    )
    chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)
    
    # Test chunk conversion to Document type
    chunk_obj.chunk_convert()
    
    # Verify converted chunks
    assert all(isinstance(chunk, Document) for chunk in chunk_obj.chunks)
    for chunk in chunk_obj.chunks:
        # Check required metadata fields
        assert 'element_id' in chunk.metadata
        assert 'type' in chunk.metadata
        assert 'chunk_size' in chunk.metadata
        assert 'chunk_overlap' in chunk.metadata
        assert 'data_source.url' not in chunk.metadata  # Because files are local
        # Check metadata types
        assert all(isinstance(v, (str, int, float, bool, list)) 
                  for v in chunk.metadata.values())

@pytest.mark.parametrize('test_index', [
    {
        'index_type': 'Pinecone',
        'embedding_service': 'OpenAI',
        'embedding_model': 'text-embedding-3-small',
        'expected_class': PineconeVectorStore
    },
    {
        'index_type': 'RAGatouille',
        'embedding_service': 'RAGatouille',
        'embedding_model': 'colbert-ir/colbertv2.0',
        'expected_class': RAGPretrainedModel
    }
])
def test_initialize_database(monkeypatch, test_index):
    """
    Test the initialization of different types of databases.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting initialize_database test with {test_index['index_type']}")
    
    index_name = 'test-index'

    # Create services
    embedding_service = EmbeddingService(
        model_service=test_index['embedding_service'],
        model=test_index['embedding_model']
    )
    
    db_service = DatabaseService(
        db_type=test_index['index_type'],
        index_name=index_name,
        embedding_service=embedding_service
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
            embedding_service=embedding_service
        )
        db_service.initialize_database(
            namespace=db_service.namespace,
            clear=True
        )

@pytest.mark.parametrize('test_index', [
    # Pinecone test
    {
        'db_types': ['Pinecone'],
        'embedding_service': 'OpenAI',
        'embedding_model': 'text-embedding-3-small',
        'expected_classes': {
            'Pinecone': PineconeVectorStore,
        }
    },
    # RAGatouille test
    {
        'db_types': ['RAGatouille'],
        'embedding_service': 'RAGatouille',
        'embedding_model': 'colbert-ir/colbertv2.0',
        'expected_classes': {
            'RAGatouille': RAGPretrainedModel
        }
    }
])
def test_delete_database(setup_fixture, test_index):
    """
    Test deleting both existing and non-existing databases for different RAG types.
    """
    logger = setup_fixture['logger']
    logger.info(f"Starting delete_database test.")
    
    index_name = 'test-delete-index'

    # Create embedding service
    embedding_service = EmbeddingService(
        model_service=test_index['embedding_service'],
        model=test_index['embedding_model']
    )
    
    # Loop through each database type and RAG type combination
    for db_type in test_index['db_types']:
        logger.info(f"\nTesting {db_type}")
        
        db_service = DatabaseService(
            db_type=db_type,
            index_name=index_name,
            embedding_service=embedding_service
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

        # Test Case 2: Create and delete database
        logger.info(f"Creating and deleting database: {db_service.index_name}")
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
            elif db_type == 'RAGatouille':
                db_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'ragatouille', db_service.index_name)
                assert not os.path.exists(db_path)

            # Verify index is not in available indexes
            available_indexes, _ = get_available_indexes(
                db_type,
                test_index['embedding_model']
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

@pytest.mark.parametrize('test_index', [
    {
        'index_type': 'Pinecone',
        'embedding_service': 'OpenAI',
        'embedding_model': 'text-embedding-3-small'
    },
    {
        'index_type': 'RAGatouille',
        'embedding_service': 'RAGatouille',
        'embedding_model': 'colbert-ir/colbertv2.0'
    }
])
def test_get_available_indexes(setup_fixture, test_index):
    """
    Test retrieving available indexes for different database and RAG configurations.
    """
    logger = setup_fixture['logger']
    logger.info(f"Starting get_available_indexes test with {test_index['index_type']}")
    
    # Create services
    embedding_service = EmbeddingService(
        model_service=test_index['embedding_service'],
        model=test_index['embedding_model']
    )

    # Create and index test documents
    index_name_initialized = f"test-{test_index['index_type'].lower()}"
    logger.info(f"Creating test index: {index_name_initialized}")
    try:
        # Initialize database service
        db_service = DatabaseService(
            db_type=test_index['index_type'],
            index_name=index_name_initialized,
            embedding_service=embedding_service
        )
        db_service.initialize_database(clear=True)

        # Initialize document processor
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            work_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'document_processing') # Put partitioned files in same directory as original files
        )
        
        # Get chunks
        partitioned_docs = doc_processor.load_partitioned_documents(
            setup_fixture['docs'],
            partition_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
        )
        chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)

        # Process and index documents
        db_service.index_data(chunk_obj)

    except Exception as e:
        logger.error(f"Error creating index {index_name_initialized}: {str(e)}")
        db_service.delete_index()
        raise e

    try:
        # Check if index is found
        logger.info(f"Testing for getting available indexes")
        try:
            expected_index = f"test-{test_index['index_type'].lower()}"

            available_indexes, index_metadatas = get_available_indexes(
                test_index['index_type'],
                test_index['embedding_model']
            )
                        
            logger.info(f"Available indexes: {available_indexes}. Checking for {expected_index}")
            
            # Verify expected index exists in results
            assert expected_index in available_indexes, \
                f"Expected index {expected_index} not found in available indexes"
            
            # Verify metadata matches
            if test_index['index_type'] != 'RAGatouille':
                for index_metadata in index_metadatas:
                    assert index_metadata['embedding_model'] == test_index['embedding_model']

        except Exception as e:
            logger.error(f"Error in specific configuration test: {str(e)}")
            raise e

    finally:
        # Clean up test index
        try:
            db_service = DatabaseService(
                db_type=test_index['index_type'],
                index_name=index_name_initialized,
                embedding_service=embedding_service,
                rerank_service=setup_fixture['mock_rerank_service']
            )
            db_service.delete_index()
            logger.info(f"Deleted test index: {index_name_initialized}")
        except Exception as e:
            logger.error(f"Error deleting index {index_name_initialized}: {str(e)}")

    
@pytest.mark.parametrize('test_index', [
    {
        'db_type': 'Pinecone',
        'embedding_service': 'OpenAI',
        'embedding_model': 'text-embedding-3-small'
    }
])
def test_index_with_different_metadatas(setup_fixture, test_index):
    """
    Test that an exception is raised when indexing documents with different chunking parameters.
    """
    # TODO figure out how to with RAGatouille eventually...
    logger = setup_fixture['logger']
    logger.info(f"Starting index_with_different_metadatas test with {test_index['db_type']}")
    
    index_name = 'test-different-params'

    # Initialize services
    embedding_service = EmbeddingService(
        model_service=test_index['embedding_service'],
        model=test_index['embedding_model']
    )

    db_service = DatabaseService(
        db_type=test_index['db_type'],
        index_name=index_name,
        embedding_service=embedding_service
    )

    try:
        # Test Case: Initialize with chunking and try to add different chunk size later
        logger.info("Test different chunk sizes in the same index")
        
        # Initialize database with chunking parameters
        initial_chunk_size = 400
        initial_chunk_overlap = 50
        
        db_service.initialize_database(clear=True)
        
        # Process and index initial documents with chunking
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            chunk_size=initial_chunk_size,
            chunk_overlap=initial_chunk_overlap
        )      

        # Process and index documents
        partitioned_docs = doc_processor.load_partitioned_documents(
            setup_fixture['docs'],
            partition_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
        )
        chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)
        db_service.index_data(chunk_obj)


        # Try to add unchunked documents (copy original doc processor and change chunk size and overlap)
        doc_processor_different_chunk_size = doc_processor
        doc_processor_different_chunk_size.chunk_size = initial_chunk_size*2
        doc_processor_different_chunk_size.chunk_overlap = initial_chunk_overlap*2
        chunk_obj_different_chunk_size, _ = doc_processor_different_chunk_size.chunk_documents(partitioned_docs)
        with pytest.raises(ValueError):
            db_service.index_data(chunk_obj_different_chunk_size)

        # Cleanup
        db_service.delete_index()
        logger.info(f'Database deleted: {test_index["db_type"]}')

    except Exception as e:
        # If there is an error, be sure to delete the database
        try:
            db_service.delete_index()
        except:
            pass
        raise e

def test_prompt_templates():
    """
    Test that prompt templates have expected input variables and correct types.
    """
    # Test CHATBOT_SYSTEM_PROMPT (SystemMessagePromptTemplate)
    assert isinstance(CHATBOT_SYSTEM_PROMPT, SystemMessagePromptTemplate)
    assert set(CHATBOT_SYSTEM_PROMPT.input_variables) == {"style_mode"}

    # Test QA_PROMPT (HumanMessagePromptTemplate)
    assert isinstance(QA_PROMPT, HumanMessagePromptTemplate)
    assert set(QA_PROMPT.input_variables) == {"context", "question"}

    # Test SUMMARIZE_TEXT (HumanMessagePromptTemplate)
    assert isinstance(SUMMARIZE_TEXT, HumanMessagePromptTemplate)
    assert set(SUMMARIZE_TEXT.input_variables) == {"augment", "summary"}

    # Test GENERATE_SIMILAR_QUESTIONS_W_CONTEXT (PromptTemplate)
    assert isinstance(GENERATE_SIMILAR_QUESTIONS_W_CONTEXT, PromptTemplate)
    assert set(GENERATE_SIMILAR_QUESTIONS_W_CONTEXT.input_variables) == {"context", "question"}

    # Test CLUSTER_LABEL (PromptTemplate)
    assert isinstance(CLUSTER_LABEL, PromptTemplate)
    assert set(CLUSTER_LABEL.input_variables) == {"documents"}

    # Test DEFAULT_DOCUMENT_PROMPT (PromptTemplate)
    assert isinstance(DEFAULT_DOCUMENT_PROMPT, PromptTemplate)
    assert set(DEFAULT_DOCUMENT_PROMPT.input_variables) == {"page_content"}

def test_prompt_validation():
    """
    Test the validation of prompts and citations in InLineCitationsResponse.
    """
    # Test valid response with correct citation format
    valid_content = """This is a test response with a valid citation <source id="1">."""
    response = InLineCitationsResponse(content=valid_content, citations=["1"])
    assert response.content == valid_content

    # Test response with no citations (should raise NoSourceCitationsFound)
    with pytest.raises(Exception) as exc_info:
        InLineCitationsResponse(content="This is a test response with no citations.", citations=[])
    assert "no source citations were found" in str(exc_info.value)

    # Test response with malformed citations
    malformed_cases = [
        "Test with space after id <source id=\"1\" >",
        "Test with no quotes <source id=1>",
        "Test with single quotes <source id='1'>",
        "Test with space in tag < source id=\"1\">",
    ]
    for case in malformed_cases:
        with pytest.raises(Exception) as exc_info:
            InLineCitationsResponse(content=case, citations=["1"])
        assert "Malformed source tags detected" in str(exc_info.value)

    # Test multiple valid citations
    multi_citation = """First citation <source id="1">. Second citation <source id="2">."""
    response = InLineCitationsResponse(content=multi_citation, citations=["1", "2"])
    assert response.content == multi_citation

def test_qa_prompt_generation(setup_fixture):
    """
    Test the generation and validation of QA prompts as used in queries.py.
    """
    logger = setup_fixture['logger']
    logger.info("Starting qa_prompt_generation test")

    llm_service = setup_fixture['mock_llm_service']

    # Create test documents with citations
    test_docs = [
        Document(
            page_content="Test content about aerospace systems.",
            metadata={"source": "test1.pdf", "page": 1}
        )
    ]

    # Format documents as they would appear in the context
    docs_content = "Source ID: 1\n" + test_docs[0].page_content + "\n\n"
    
    # Test question
    test_question = "What can you tell me about aerospace systems?"

    # Generate the prompt
    system_prompt = CHATBOT_SYSTEM_PROMPT.format(style_mode=None)
    prompt_with_context = QA_PROMPT.format(
        context=docs_content,
        question=test_question
    )
    messages = [system_prompt, prompt_with_context]

    # Get response from LLM
    response = llm_service.get_llm().invoke(messages)
    
    try:
        # Parse response - should contain citations
        parsed_response = PydanticOutputParser(pydantic_object=InLineCitationsResponse).parse(response.content)
        assert isinstance(parsed_response, InLineCitationsResponse)
        assert "<source id=\"1\">" in parsed_response.content
        assert "1" in parsed_response.citations
    except Exception as e:
        logger.error(f"Failed to parse response: {response.content}")
        raise e

    # Test with empty context
    empty_prompt = QA_PROMPT.format(
        context="",
        question=test_question
    )
    messages = [system_prompt, empty_prompt]
    response = llm_service.get_llm().invoke(messages)

    with pytest.raises(Exception) as exc_info:
        PydanticOutputParser(pydantic_object=InLineCitationsResponse).parse(response.content)
    assert "no source citations were found" in str(exc_info.value)  # TODO use custom exception

def test_database_setup_and_query(test_input, setup_fixture):
    """
    Tests the entire process of initializing a database, upserting documents, and deleting a database.
    """
    logger = setup_fixture['logger']
    test, print_str = parse_test_case(setup_fixture, test_input)
    logger.info(f"Starting database_setup_and_query test: {print_str}")

    from aerospace_chatbot.services.database import DatabaseService
    from aerospace_chatbot.processing import DocumentProcessor

    index_name = 'test' + str(test['id'])
    logger.info(f'Starting test: {print_str}')

    # Get services
    embedding_service = EmbeddingService(
        model_service=test['embedding_service'],
        model=test['embedding_model']
    )
    
    if test['rerank_service'] is not None:
        rerank_service = RerankService(
            model_service=test['rerank_service'],
            model=test['rerank_model']
        )
    else:
        rerank_service = None

    llm_service = LLMService(
        model_service=test['llm_service'],
        model=test['llm_model']
    )

    db_service = DatabaseService(
        db_type=test['index_type'],
        index_name=index_name,
        embedding_service=embedding_service,
        rerank_service=rerank_service
    )

    try:
        # Initialize the document processor with services
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap']
        )
        # Process and index documents
        db_service.initialize_database(
            clear=True
        )
        # Verify the vectorstore type
        if db_service.db_type == 'Pinecone':
            assert isinstance(db_service.vectorstore, PineconeVectorStore)
        elif db_service.db_type == 'RAGatouille':
            assert isinstance(db_service.vectorstore, RAGPretrainedModel)
        logger.info('Vectorstore created.')

        # Process and index documents
        partitioned_docs = doc_processor.load_partitioned_documents(
            setup_fixture['docs'],
            partition_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
        )
        chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)
        db_service.index_data(chunk_obj)

        # Initialize QA model
        qa_model = QAModel(
            db_service=db_service,
            llm_service=llm_service
        )
        logger.info('QA model object created.')
        assert qa_model is not None

        # Run a query and verify results
        qa_model.query(setup_fixture['test_prompt'])
        assert qa_model.result['messages'] is not None
        assert qa_model.result['context'] is not None
        assert qa_model.result['alternative_questions'] is not None

        # Delete the indexes
        db_service.delete_index()
        
        logger.info('Databases deleted.')

    except Exception as e:  # If there is an error, be sure to delete the database
        db_service.delete_index()
        raise e

@pytest.mark.parametrize('test_index', [
    {
        'db_type': 'Pinecone',
        'embedding_service': 'OpenAI',
        'embedding_model': 'text-embedding-3-small'
    }
])
def test_get_docs_df(setup_fixture, test_index):
    """
    Test function for the get_docs_df() method.
    """
    logger = setup_fixture['logger']
    logger.info(f"Starting get_docs_df test with {test_index['db_type']}")
    
    index_name = 'test-viz-df-export'

    # Initialize services
    embedding_service = EmbeddingService(
        model_service=test_index['embedding_service'],
        model=test_index['embedding_model']
    )
    db_service = DatabaseService(
        db_type=test_index['db_type'],
        index_name=index_name,
        embedding_service=embedding_service
    )
    db_service.initialize_database(clear=True)

    try:
        # Initialize the document processor with services
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap']
        )

        # Process and index documents
        partitioned_docs = doc_processor.load_partitioned_documents(
            setup_fixture['docs'],
            partition_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
        )
        chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)
        db_service.index_data(chunk_obj)

        # Get combined dataframe
        df = get_docs_df(
            db_service=db_service,  # Main document database service
        )

        # Assert the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in [
            "id", "page_content", "page_number", "file_name", "embedding", "type",
        ])

        # Verify there is at least one document and one question
        assert len(df[df['type'] == 'doc']) > 0, "No documents found in DataFrame"
        
        # Cleanup
        db_service.delete_index()
        logger.info(f'Database deleted: {test_index["db_type"]}')

    except Exception as e:  # If there is an error, be sure to delete the database
        try:
            db_service.delete_index()
        except:
            pass
        raise e
    
@pytest.mark.parametrize('test_index', [
    {
        'db_type': 'Pinecone',
        'embedding_service': 'OpenAI',
        'embedding_model': 'text-embedding-3-small'
    }
])
def test_add_clusters(setup_fixture, test_index):
    """
    Test function for the add_clusters function.
    """
    logger = setup_fixture['logger']
    logger.info(f"Starting add_clusters test with {test_index['db_type']}")
    
    index_name = 'test-visualization-add-clusters'

    # Initialize services
    embedding_service = EmbeddingService(
        model_service=test_index['embedding_service'],
        model=test_index['embedding_model']
    )
    llm_service = LLMService(
        model_service='OpenAI',
        model='gpt-4o-mini'
    )
    db_service = DatabaseService(
        db_type=test_index['db_type'],
        index_name=index_name,
        embedding_service=embedding_service,
        rerank_service=setup_fixture['mock_rerank_service']
    )

    try:
        # Initialize the document processor with services
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap']
        )

        # Process and index documents
        db_service.initialize_database(
            clear=True
        )
        partitioned_docs = doc_processor.load_partitioned_documents(
            setup_fixture['docs'],
            partition_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
        )
        chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)
        db_service.index_data(chunk_obj)

        # Get combined dataframe
        df = get_docs_df(
            db_service=db_service,  # Main document database service
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
        logger.info(f'Database deleted: {test_index["db_type"]}')

    except Exception as e:  # If there is an error, be sure to delete the database
        try:
            db_service.delete_index()
        except:
            pass
        raise e

# FIXME This test is failing because the process_uploads function is not implemented right now
# def test_process_user_doc_uploads(setup_fixture):
#     """
#     Test processing and merging user uploads into existing database.
#     """
#     logger = setup_fixture['logger']
#     logger.info("Starting process_user_doc_uploads test")

#     # Create mock sidebar settings
#     mock_sb = {
#         'index_type': 'Pinecone',
#         'index_selected': 'test-process-uploads',
#         'embedding_service': 'OpenAI',
#         'embedding_model': 'text-embedding-3-small',
#     }
#     upsert_docs = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '1999_honnen_reocr.pdf')]

#     try:
#         # First create and populate initial database with setup docs
#         embedding_service = EmbeddingService(
#             model_service=mock_sb['embedding_service'],
#             model=mock_sb['embedding_model']
#         )
        
#         db_service = DatabaseService(
#             db_type=mock_sb['index_type'],
#             index_name=mock_sb['index_selected'],
#             embedding_service=embedding_service,
#             rerank_service=setup_fixture['mock_rerank_service']
#         )

#         # Initialize database and index setup docs
#         db_service.initialize_database(clear=True)
        
#         doc_processor = DocumentProcessor(
#             embedding_service=embedding_service,
#             chunk_size=setup_fixture['chunk_size'],
#             chunk_overlap=setup_fixture['chunk_overlap']
#         )
        
#         # Process and index initial documents
#         partitioned_docs = doc_processor.load_partitioned_documents(
#             setup_fixture['docs'],
#             partition_dir=os.path.join(os.path.dirname(setup_fixture['docs'][0]), 'test_processed_docs')
#         )
#         chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)
#         db_service.index_data(chunk_obj) 
        
#         # Process the "uploaded" docs
#         user_upload = process_uploads(mock_sb, upsert_docs)
#         logger.info(f"User upload from process_uploads: {user_upload}")

#         # Verify documents were indexed in new namespace
#         pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
#         index = pc.Index(mock_sb['index_selected'])
#         stats = index.describe_index_stats()
#         logger.info(f"Index stats: {stats}")
#         assert stats['namespaces'][user_upload]['vector_count'] > 0
        
#         # Cleanup
#         db_service.delete_index()

#     except Exception as e:
#         # Ensure cleanup on failure
#         try:
#             db_service.delete_index()
#         except:
#             pass
#         raise e

### Frontend tests
def test_sidebar_manager(setup_fixture):
    """
    Test the SidebarManager class functionality.
    """
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

    # Test single case since render_sidebar now renders everything
    sb_out = sidebar_manager.render_sidebar()
    
    # Verify all outputs are present since everything is rendered
    # Core dependencies
    assert 'index_type' in sb_out
    
    # Embeddings outputs
    assert 'embedding_service' in sb_out
    assert 'embedding_model' in sb_out
        
    # LLM outputs
    assert 'llm_service' in sb_out
    assert 'llm_model' in sb_out
    
    # Model options outputs
    assert 'model_options' in sb_out
    assert 'temperature' in sb_out['model_options']
    assert 'output_level' in sb_out['model_options']
    assert 'k_retrieve' in sb_out['model_options']
    assert 'k_rerank' in sb_out['model_options']
    assert 'style_mode' in sb_out['model_options']

def test_sidebar_manager_invalid_config(setup_fixture):
    """
    Test SidebarManager initialization with invalid config file path
    """
    logger = setup_fixture['logger']
    logger.info("Starting sidebar_manager_invalid_config test")
    
    with pytest.raises(ConfigurationError):
        SidebarManager('nonexistent_config.json')

def test_sidebar_manager_malformed_config(setup_fixture):
    """
    Test SidebarManager with malformed config file
    """
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
    """
    Test SidebarManager with config missing required sections
    """
    logger = setup_fixture['logger']
    logger.info("Starting sidebar_manager_missing_required_sections test")
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tf:
        # Create config missing 'databases' section
        json.dump({'embeddings': {}, 'llms': {}}, tf)
        tf.flush()
        with pytest.raises(ConfigurationError):
            SidebarManager(tf.name)

def test_set_secrets_with_valid_input(setup_fixture):
    """
    Test case for set_secrets function with valid input.
    """
    logger = setup_fixture['logger']
    logger.info("Starting set_secrets_with_valid_input test")
    
    test_secrets = {
        'OPENAI_API_KEY': 'openai_key',
        'VOYAGE_API_KEY': 'voyage_key',
        'PINECONE_API_KEY': 'pinecone_key',
        'HUGGINGFACEHUB_API_KEY': 'huggingface_key',
        'ANTHROPIC_API_KEY': 'anthropic_key',
        'UNSTRUCTURED_API_KEY': 'unstructured_key',
        'UNSTRUCTURED_API_URL': 'unstructured_url'
    }
    
    # Set secrets and verify return
    result = set_secrets(test_secrets)
    assert result == test_secrets
    
    # Verify environment variables were set
    for key, value in test_secrets.items():
        assert os.environ[key] == value