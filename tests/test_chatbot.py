import os, sys, json
import itertools
import pytest
import pandas as pd
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
    cache_resource,
    ConfigurationError,
    get_cache_decorator, 
    get_cache_data_decorator,
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

# TODO add tests to check conversation history functionality
# TODO add a test to check parent/child and summary lookup functionality (not just that it executes)
# TODO add upload file test
# TODO add ap test from streamlit
# TODO test retrieval of metadata vector from databases

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
        'query_model': test_case['query_model'],
        'embedding_name': test_case['embedding_name'],
        'rag_type': setup['rag_type'][test_case['rag_type']],
        'llm_family': test_case['llm_family'],
        'llm': test_case['llm']
    }
    print_str = ', '.join(f'{key}: {value}' for key, value in test_case.items())

    return parsed_test, print_str
def parse_test_model(type, test):
    """Parses the test model based on the given type and test parameters."""
    if type == 'embedding':
        # Initialize the embedding service
        if test['query_model'] == 'OpenAI':
            embedding_service = EmbeddingService(
                model_name=test['embedding_name'],
                model_type='OpenAI'
            )
        elif test['query_model'] == 'Voyage':
            embedding_service = EmbeddingService(
                model_name=test['embedding_name'],
                model_type='Voyage'
            )
        elif test['query_model'] == 'Hugging Face':
            embedding_service = EmbeddingService(
                model_name=test['embedding_name'],
                model_type='Hugging Face'
            )
        else:
            raise NotImplementedError('Query model not implemented.')
        return embedding_service

    elif type == 'llm':
        # Initialize the LLM service
        if test['llm_family'] == 'OpenAI':
            llm_service = LLMService(
                model_name=test['llm'],
                model_type='OpenAI'
            )
        elif test['llm_family'] == 'Anthropic':
            llm_service = LLMService(
                model_name=test['llm'],
                model_type='Anthropic'
            )
        elif test['llm_family'] == 'Hugging Face':
            llm_service = LLMService(
                model_name=test['llm'],
                model_type='Hugging Face'
            )
        else:
            raise NotImplementedError('LLM not implemented.')
        return llm_service

    else:
        raise ValueError('Invalid type. Must be either "embedding" or "llm".')

# Fixtures
@pytest.fixture(scope='session', autouse=True)
def setup_fixture():
    """Sets up the fixture for testing the backend."""
    # Pull api keys from .env file. If these do not exist, create a .env file in the root directory and add the following.
    load_dotenv(find_dotenv(), override=True)
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
        model_name='text-embedding-3-large',
        model_type='OpenAI'
    )

    mock_llm_service = LLMService(
        model_name='gpt-4o-mini',
        model_type='OpenAI',
        temperature=0,
        max_tokens=1000
    )

    setup = {
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
@pytest.fixture()
def temp_dotenv(setup_fixture):
    """Creates a temporary .env file for testing purposes."""
    dotenv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '.env')
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            print('Creating .env file for testing.')
            f.write(f'OPENAI_API_KEY = {setup_fixture["OPENAI_API_KEY"]}\n')
            f.write(f'PINECONE_API_KEY = {setup_fixture["PINECONE_API_KEY"]}\n')
            f.write(f'HUGGINGFACEHUB_API_KEY = {setup_fixture["HUGGINGFACEHUB_API_KEY"]}\n')
            f.write(f'LOCAL_DB_PATH = {setup_fixture["LOCAL_DB_PATH"]}\n')
        yield dotenv_path
        os.remove(dotenv_path)
    else:
        yield dotenv_path

### Begin tests
def test_validate_index(setup_fixture):
    """Test edge cases for validate_index function."""
    from aerospace_chatbot.services.database import DatabaseService
    from aerospace_chatbot.processing import DocumentProcessor
    
    # Test case 1: Empty index name
    db_type='ChromaDB'
    db_service = DatabaseService(
        db_type=db_type,
        index_name='',
        rag_type='Standard'
    )

    doc_processor = DocumentProcessor(
        embedding_service=setup_fixture['mock_embedding_service'],
        rag_type='Standard',
        chunk_size=400,
        chunk_overlap=50
    )
    
    with pytest.raises(ValueError, match="Index name cannot be empty"):
        db_service.index_name = ""
        db_service._validate_index(doc_processor)

    # Test case 2: Whitespace-only index name
    with pytest.raises(ValueError, match="Index name cannot be empty"):
        db_service.index_name = "   "
        db_service._validate_index(doc_processor)

    # Test case 3: ChromaDB invalid characters
    with pytest.raises(ValueError, match="can only contain alphanumeric characters"):
        db_service.index_name = "test@index"
        db_service._validate_index(doc_processor)

    # Test case 4: ChromaDB consecutive periods
    with pytest.raises(ValueError, match="can only contain alphanumeric characters, underscores, or hyphens"):
        db_service.index_name = "test..index"
        db_service._validate_index(doc_processor)

    # Test case 5: ChromaDB non-alphanumeric start/end
    with pytest.raises(ValueError, match="must start and end with an alphanumeric character"):
        db_service.index_name = "-testindex-"
        db_service._validate_index(doc_processor)

    # Test case 6: ChromaDB name too long
    with pytest.raises(ValueError, match="must be less than 63 characters"):
        db_service.index_name = "a" * 64
        db_service._validate_index(doc_processor)

    # Test case 7: Pinecone name too long
    db_type='Pinecone'
    db_service = DatabaseService(
        db_type=db_type,
        index_name='',
        rag_type='Standard'
    )

    with pytest.raises(ValueError, match="must be less than 45 characters"):
        db_service.index_name = "a" * 46
        db_service._validate_index(doc_processor)

    # Test case 8: Summary RAG type without LLM service
    doc_processor.rag_type = "Summary"
    db_service.rag_type = doc_processor.rag_type
    doc_processor.llm_service = None
    with pytest.raises(ValueError, match="LLM service is required for Summary RAG type"):
        db_service.index_name = "test-index"
        db_service._validate_index(doc_processor)

    # Test case 9: Valid cases with different RAG types
    db_type='ChromaDB'
    index_name = "test-index"

    # Standard RAG
    db_service = DatabaseService(
        db_type=db_type,
        index_name=index_name,
        rag_type='Standard'
    )
    doc_processor.rag_type = db_service.rag_type
    db_service._validate_index(doc_processor)
    assert db_service.index_name == index_name
    
    # Parent-Child RAG
    db_service = DatabaseService(
        db_type=db_type,
        index_name=index_name,
        rag_type='Parent-Child'
    )
    doc_processor.rag_type = db_service.rag_type
    db_service._validate_index(doc_processor)
    assert db_service.index_name == index_name + "-parent-child"
    
    # Summary RAG
    db_service = DatabaseService(
        db_type=db_type,
        index_name=index_name,
        rag_type='Summary'
    )
    doc_processor.rag_type = db_service.rag_type
    doc_processor.llm_service = setup_fixture['mock_llm_service']
    db_service._validate_index(doc_processor)
    assert db_service.index_name == index_name + "-summary"

# Test chunk docs
def test_process_documents_standard(setup_fixture):
    '''Test document processing with standard RAG.'''
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
        'query_model': 'OpenAI',
        'embedding_name': 'text-embedding-3-large',
        'expected_class': PineconeVectorStore
    },
    {
        'index_type': 'ChromaDB',
        'query_model': 'OpenAI',
        'embedding_name': 'text-embedding-ada-002',
        'expected_class': Chroma
    },
    {
        'index_type': 'RAGatouille',
        'query_model': 'RAGatouille',
        'embedding_name': 'colbert-ir/colbertv2.0',
        'expected_class': RAGPretrainedModel
    }
])
def test_initialize_database(monkeypatch, setup_fixture, test_index):
    '''Test the initialization of different types of databases.'''
    # FIXME, work through where rag_type should be defined, it's a little messy at the moment. Probably belongs in the initialization of DatabaseService.
    index_name = 'test-index'
    rag_type = 'Standard'

    # Create services
    embedding_service = EmbeddingService(
        model_name=test_index['embedding_name'],
        model_type=test_index['query_model']
    )
    
    db_service = DatabaseService(
        db_type=test_index['index_type'],
        index_name=index_name,
        rag_type=rag_type
    )

    # Clean up any existing database first
    try:
        db_service.delete_index()
    except:
        pass  # Ignore errors if database doesn't exist

    # Test with environment variable local_db_path
    try:
        vectorstore = db_service.initialize_database(
            embedding_service=embedding_service,
            namespace=db_service.namespace,
            clear=True
        )

        assert isinstance(vectorstore, test_index['expected_class'])
        
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
            rag_type=rag_type
        )
        db_service.initialize_database(
            embedding_service=embedding_service,
            namespace=db_service.namespace,
            clear=True
        )
@pytest.mark.parametrize('test_index', [
    {
        'index_type': 'Pinecone',
        'query_model': 'OpenAI',
        'embedding_name': 'text-embedding-3-large',
        'expected_class': PineconeVectorStore
    },
    {
        'index_type': 'ChromaDB',
        'query_model': 'OpenAI',
        'embedding_name': 'text-embedding-ada-002',
        'expected_class': Chroma
    },
    {
        'index_type': 'RAGatouille',
        'query_model': 'RAGatouille',
        'embedding_name': 'colbert-ir/colbertv2.0',
        'expected_class': RAGPretrainedModel
    }
])
def test_delete_database(setup_fixture, test_index):
    '''Test deleting both existing and non-existing databases.'''
    # FIXME check that local filestores are deleted
    index_name = 'test-delete-index'
    rag_type = 'Standard'

    # Create services
    embedding_service = EmbeddingService(
        model_name=test_index['embedding_name'],
        model_type=test_index['query_model']
    )
    
    db_service = DatabaseService(
        db_type=test_index['index_type'],
        index_name=index_name,
        rag_type=rag_type
    )

    # Clean up any existing test indexes first
    try:
        db_service.delete_index()
    except Exception as e:
        print(f"Info: Cleanup of existing index failed (this is expected if index didn't exist): {str(e)}")

    # Test Case 1: Delete non-existent database
    try:
        db_service.delete_index()
    except Exception as e:
        assert "does not exist" in str(e).lower() or "not found" in str(e).lower()

    # Test Case 2: Create and delete standard database
    try:
        vectorstore = db_service.initialize_database(
            embedding_service=embedding_service,
            namespace=db_service.namespace,
            clear=True
        )
        assert isinstance(vectorstore, test_index['expected_class'])
        
        # Delete the database
        db_service.delete_index()
        
        # Verify deletion by checking if database exists
        if test_index['index_type'] == 'Pinecone':
            pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            assert index_name not in pc.list_indexes()
        elif test_index['index_type'] == 'ChromaDB':
            db_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], index_name)
            assert not os.path.exists(db_path)
        elif test_index['index_type'] == 'RAGatouille':
            db_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'ragatouille', index_name)
            assert not os.path.exists(db_path)

    except Exception as e:
        # If test fails, ensure cleanup
        try:
            db_service.delete_index()
        except:
            pass
        raise e
    
# Test end to end process, adding query
def test_database_setup_and_query(test_input, setup_fixture):
    '''Tests the entire process of initializing a database, upserting documents, and deleting a database.'''
    from aerospace_chatbot.services.database import DatabaseService
    from aerospace_chatbot.processing import DocumentProcessor

    test, print_str = parse_test_case(setup_fixture, test_input)
    index_name = 'test' + str(test['id'])
    print(f'Starting test: {print_str}')

    # Get services
    db_service = DatabaseService(
        db_type=test['index_type'],
        index_name=index_name,
        rag_type=test['rag_type']
    )
    query_model_service = parse_test_model('embedding', test)
    llm_service = parse_test_model('llm', test)

    # Print query model service details
    print("\nQuery Model Service Details:")
    print(f"Model Name: {query_model_service.model_name}")
    print(f"Model Type: {query_model_service.model_type}")
    print(f"Embedding Dimension in test: {query_model_service.get_dimension()}")

    try:
        # Initialize the document processor with services
        doc_processor = DocumentProcessor(
            embedding_service=query_model_service,
            rag_type=test['rag_type'],
            chunk_method=setup_fixture['chunk_method'],
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            llm_service=llm_service
        )

        # Process and index documents
        chunking_result = doc_processor.process_documents(setup_fixture['docs'])
        db_service.index_documents(
            chunking_result=chunking_result,
            batch_size=setup_fixture['batch_size'],
            clear=True
        )

        # Verify the vectorstore type
        if db_service.db_type == 'ChromaDB':
            assert isinstance(doc_processor.vectorstore, Chroma)
        elif db_service.db_type == 'Pinecone':
            assert isinstance(doc_processor.vectorstore, PineconeVectorStore)
        elif db_service.db_type == 'RAGatouille':
            assert isinstance(doc_processor.vectorstore, RAGPretrainedModel)
        print('Vectorstore created.')

        # Initialize QA model
        qa_model = QAModel(
            db_service=db_service,
            llm_service=llm_service
        )
        print('QA model object created.')
        assert qa_model is not None

        # Run a query and verify results
        qa_model.query(setup_fixture['test_prompt'])
        assert qa_model.ai_response is not None
        assert qa_model.sources is not None
        assert qa_model.memory is not None

        # Generate alternative questions
        alternate_question = qa_model.generate_alternative_questions(setup_fixture['test_prompt'])
        assert alternate_question is not None
        print('Query and alternative question successful!')

        # Delete the index
        db_service.delete_index()
        if doc_processor.rag_type in ['Parent-Child', 'Summary']:
            lfs_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'local_file_store', index_name)
            assert not os.path.exists(lfs_path)  # Check that the local file store was deleted
        print('Database deleted.')

    except Exception as e:  # If there is an error, be sure to delete the database
        db_service.delete_index()
        raise e

# Test sidebar loading and secret keys
def test_sidebar_manager():
    """Test the SidebarManager class functionality."""
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
    assert 'query_model' in sb_out
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
    if sb_out['index_type'] != 'RAGatouille':
        assert 'search_type' in sb_out['model_options']

    # Test paths functionality
    paths = sidebar_manager.get_paths(home_dir)
    assert 'base_folder_path' in paths
    assert 'db_folder_path' in paths
def test_sidebar_manager_invalid_config():
    """Test SidebarManager initialization with invalid config file path"""
    with pytest.raises(FileNotFoundError):
        SidebarManager('nonexistent_config.json')
def test_sidebar_manager_malformed_config():
    """Test SidebarManager with malformed config file"""
    # Create a temporary malformed config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tf:
        tf.write('{"invalid": "json"')
        tf.flush()
        with pytest.raises(json.JSONDecodeError):
            SidebarManager(tf.name)
def test_sidebar_manager_missing_required_sections():
    """Test SidebarManager with config missing required sections"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tf:
        # Create config missing 'databases' section
        json.dump({'embeddings': {}, 'llms': {}, 'rag_types': {}}, tf)
        tf.flush()
        with pytest.raises(KeyError):
            SidebarManager(tf.name)
def test_sidebar_manager_get_paths():
    """Test get_paths method with various inputs"""
    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file = os.path.join(home_dir, 'config', 'config_admin.json')
    
    manager = SidebarManager(config_file)
    
    # Test with valid path
    paths = manager.get_paths(home_dir)
    assert os.path.exists(paths['base_folder_path'])
    assert os.path.exists(paths['db_folder_path'])
    
    # Test with invalid path
    with pytest.raises(Exception):
        manager.get_paths('/nonexistent/path')
    
    # Test with empty path
    with pytest.raises(Exception):
        manager.get_paths('')
def test_sidebar_manager_validate_config():
    """Test _validate_config method with various inputs"""
    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file = os.path.join(home_dir, 'config', 'config_admin.json')
    
    manager = SidebarManager(config_file)
    
    # Test with valid config
    valid_config = {
        'databases': {'test': {}},
        'embeddings': {'test': {}},
        'llms': {'test': {}},
        'rag_types': {'test': {}}
    }
    assert manager._validate_config(valid_config) is None
    
    # Test with missing sections
    invalid_config = {'databases': {}}
    with pytest.raises(Exception):
        manager._validate_config(invalid_config)
    
    # Test with empty config
    with pytest.raises(Exception):
        manager._validate_config({})

def test_set_secrets_with_valid_input():
    '''Test case for set_secrets function with valid input.'''
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

def test_get_secrets_with_dotenv(tmp_path, monkeypatch):
    '''Test get_secrets with .env file'''
    # Create temporary .env file
    env_path = tmp_path / '.env'
    env_content = '\n'.join([
        'OPENAI_API_KEY=openai_key',
        'VOYAGE_API_KEY=voyage_key',
        'PINECONE_API_KEY=pinecone_key',
        'HUGGINGFACEHUB_API_KEY=huggingface_key',
        'ANTHROPIC_API_KEY=anthropic_key'
    ])
    env_path.write_text(env_content)
    
    # Temporarily change working directory
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        secrets = get_secrets()
        
        # Verify secrets were loaded from .env
        assert secrets['OPENAI_API_KEY'] == 'openai_key'
        assert secrets['VOYAGE_API_KEY'] == 'voyage_key'
        assert secrets['PINECONE_API_KEY'] == 'pinecone_key'
        assert secrets['HUGGINGFACEHUB_API_KEY'] == 'huggingface_key'
        assert secrets['ANTHROPIC_API_KEY'] == 'anthropic_key'

def test_get_docs_questions_df(setup_fixture):
    """Test function for the get_docs_questions_df() method."""
    index_name = 'test-visualization'
    
    # Initialize services
    db_service = DatabaseService(
        db_type='ChromaDB',
        index_name=index_name,
        rag_type='Standard'
    )
    embedding_service = EmbeddingService(
        model_name='text-embedding-3-large',
        model_type='OpenAI'
    )
    llm_service = LLMService(
        model_name='gpt-4o-mini',
        model_type='OpenAI'
    )

    try:
        # Initialize document processor
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            llm_service=llm_service,
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap']
        )

        # Process and index documents
        doc_processor.process_and_index(
            documents=setup_fixture['docs'],
            index_name=index_name,
            clear=True
        )

        # Create QA model and run query
        qa_model = QAModel(
            db_service=db_service,
            embedding_service=embedding_service,
            llm_service=llm_service
        )
        qa_model.query_docs(setup_fixture['test_prompt'])

        # Get combined dataframe
        df = DatabaseService.get_docs_questions_df(index_name, index_name+'-queries', embedding_service)

        # Assert the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in [
            "id", "source", "page", "document", "embedding", "type",
            "first_source", "used_by_questions", "used_by_num_questions",
            "used_by_question_first"
        ])

        # Cleanup
        db_service.delete_index()

    except Exception as e:
        db_service.delete_index()
        raise e

def test_add_clusters(setup_fixture):
    """Test function for the add_clusters function."""
    # Setup same as test_get_docs_questions_df
    index_name = 'test-visualization'
    db_service = DatabaseService(
        db_type='ChromaDB',
        index_name=index_name,
        rag_type='Standard' 
    )
    embedding_service = EmbeddingService(
        model_name='text-embedding-3-large',
        model_type='OpenAI'
    )
    llm_service = LLMService(
        model_name='gpt-4o-mini',
        model_type='OpenAI'
    )

    try:
        # Initialize and process documents (same as previous test)
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            llm_service=llm_service,
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap']
        )
        doc_processor.process_and_index(
            documents=setup_fixture['docs'],
            index_name=index_name,
            clear=True
        )

        qa_model = QAModel(
            db_service=db_service,
            embedding_service=embedding_service,
            llm_service=llm_service
        )
        qa_model.query_docs(setup_fixture['test_prompt'])

        df = DatabaseService.get_docs_questions_df(index_name, index_name+'-queries', embedding_service)
        db_service.delete_index()

    except Exception as e:
        db_service.delete_index()
        pytest.fail(f"Test failed with exception: {str(e)}")

    # Test clustering without labels
    n_clusters = 2
    df_with_clusters = DatabaseService.add_clusters(df, n_clusters)
    assert len(df_with_clusters["Cluster"].unique()) == n_clusters
    for cluster in df_with_clusters["Cluster"].unique():
        assert len(df_with_clusters[df_with_clusters["Cluster"] == cluster]) >= 1

    # Test clustering with labels
    df_with_clusters = DatabaseService.add_clusters(df, n_clusters, llm_service, 2)
    assert len(df_with_clusters["Cluster"].unique()) == n_clusters
    assert "Cluster_Label" in df_with_clusters.columns
    assert df_with_clusters["Cluster_Label"].notnull().all()
    assert df_with_clusters["Cluster_Label"].apply(lambda x: isinstance(x, str)).all()
    for cluster in df_with_clusters["Cluster"].unique():
        assert len(df_with_clusters[df_with_clusters["Cluster"] == cluster]) > 0