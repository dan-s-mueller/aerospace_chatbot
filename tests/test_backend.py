import os, sys, json
import itertools
import pytest
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma

from langchain_voyageai import VoyageAIEmbeddings

from ragatouille import RAGPretrainedModel

import chromadb
from chromadb import ClientAPI

from ragxplorer import RAGxplorer

# Import local variables
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src/aerospace_chatbot'))
from data_processing import chunk_docs, initialize_database, load_docs, delete_index, reduce_vector_query_size, create_data_viz
from admin import load_sidebar, set_secrets, st_setup_page, SecretKeyException
from queries import QA_Model

# Functions
def permute_tests(test_data):
    """
    Generate permutations of test cases.

    Args:
        test_data (list): List of dictionaries containing test case data.

    Returns:
        list: List of dictionaries representing permuted test cases.
    """
    rows = []
    idx=0
    for row_data in test_data:
        keys = row_data.keys()
        values = row_data.values()
        permutations = list(itertools.product(*values))
        for perm in permutations:
            row = dict(zip(keys, perm))
            row['id'] = idx
            rows.append(row)
            idx+=1
    return rows
def generate_test_cases(export:bool=True,export_dir:str='.'):
    '''
    Generate test cases and export them to a JSON file.

    Args:
        export (bool, optional): Whether to export the test cases to a JSON file. Defaults to True.
        export_dir (str, optional): Directory to export the JSON file. Defaults to '.'.

    Returns:
        list: List of dictionaries representing the generated test cases.
    '''
    # Items in test_cases must match labels to select from in setup_fixture
    # TODO throw in bad inputs for each of the 4 major types below.
    test_cases = [
        {
            # Tests ChromaDB setups, advanced RAG (standard/parent-child)
            'index_type': ['ChromaDB'],
            'query_model': ['OpenAI'],
            'rag_type': ['Standard', 'Parent-Child'],
            'llm': ['Hugging Face']
        },
        {
            # Tests advanced RAG (summary) and LLM (openai/hugging face)
            'index_type': ['ChromaDB'],
            'query_model': ['OpenAI'],
            'rag_type': ['Summary'],
            'llm': ['Hugging Face','OpenAI']
        },
        {
            # Tests Pinecone setups, embedding types (openai/voyage)
            'index_type': ['Pinecone'],
            'query_model': ['OpenAI', 'Voyage'],
            'rag_type': ['Standard'],
            'llm': ['Hugging Face']
        },
        {
            # Tests RAGatouille setup
            'index_type': ['RAGatouille'],
            'query_model': ['RAGatouille'],
            'rag_type': ['Standard'],
            'llm': ['Hugging Face']
        }
    ]

    tests = permute_tests(test_cases)
    
    if export:
        file_path = os.path.join(export_dir,'test_cases.json')
        with open(file_path, 'w') as json_file:
            json.dump(tests, json_file, indent=4)
    
    return tests
def read_test_cases(json_path:str):
    '''
    Read test cases from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: List of dictionaries representing the test cases.
    '''
    with open(json_path, 'r') as json_file:
        test_cases = json.load(json_file)
    return test_cases
def pytest_generate_tests(metafunc):
    '''
    Use pytest_generate_tests to dynamically generate tests.
    Tests generates tests from a static file (test_cases.json). You must run generate_test_cases() first.

    Args:
        metafunc: The metafunc object provided by pytest.
    '''
    if "test_input" in metafunc.fixturenames:
        tests = read_test_cases(os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_cases.json'))
        metafunc.parametrize("test_input", tests)
def parse_test_case(setup,test_case):
    ''' 
    Parse test case to be used in the test functions.

    Args:
        setup (dict): The setup variables and configurations.
        test_case (dict): The test case data.

    Returns:
        tuple: A tuple containing the parsed test case and a string representation of the test case.
    '''
    parsed_test = {
        'id': test_case['id'],
        'index_type': setup['index_type'][test_case['index_type']],
        'query_model': setup['query_model'][test_case['query_model']],
        'rag_type': setup['rag_type'][test_case['rag_type']],
        'llm': setup['llm'][test_case['llm']],
    }
    print_str = ', '.join(f"{key}: {value}" for key, value in test_case.items())

    return parsed_test, print_str
def viz_database_setup(index_name:str,setup:dict):
    '''
    Set up the RAGxplorer and ChromaDB for database visualization.

    Args:
        index_name (str): Name of the index.
        setup (dict): The setup variables and configurations.

    Returns:
        tuple: A tuple containing the RAGxplorer client and ChromaDB client.
    '''
    rx_client = RAGxplorer(embedding_model='text-embedding-ada-002')
    chroma_client = chromadb.PersistentClient(path=os.path.join(setup['LOCAL_DB_PATH'],'chromadb'))

    # Initialize a small database
    index_type = 'ChromaDB'
    rag_type = 'Standard'
    try:
        vectorstore = load_docs(
            'ChromaDB',
            setup['docs'],
            rag_type=rag_type,
            query_model=setup['query_model']['OpenAI'],
            index_name=index_name, 
            chunk_size=setup['chunk_size']*2,   # Double chunk size to reduce quantity
            chunk_overlap=setup['chunk_overlap'],
            clear=True,
            batch_size=setup['batch_size'],
            local_db_path=setup['LOCAL_DB_PATH'])
    except Exception as e:
        delete_index(index_type, index_name, rag_type, local_db_path=setup['LOCAL_DB_PATH'])
    return rx_client, chroma_client

# Fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_fixture():
    """
    Sets up the necessary variables and configurations for the test.
    The tests in this script will only work if there exists environment variables for API keys: 
    OPENAI_API_KEY, VOYAGE_API_KEY, HUGGINGFACEHUB_API_TOKEN, and PINECONE_API_KEY.

    Returns:
        dict: A dictionary containing the setup variables and configurations.
    """    
    # Pull api keys from .env file. If these do not exist, create a .env file in the root directory and add the following.
    load_dotenv(find_dotenv(),override=True)
    OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
    VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
    HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

    # Set environment variables from .env file. They are required for items tested here. This is done in the GUI setup.
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    os.environ['VOYAGE_API_KEY'] = VOYAGE_API_KEY
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

    LOCAL_DB_PATH=os.path.abspath(os.path.dirname(__file__))   # Default to the test path for easy cleanup.
    # Set default to environment variable
    os.environ['LOCAL_DB_PATH'] = LOCAL_DB_PATH
    
    # Fixed inputs
    docs = ['test1.pdf', 'test2.pdf']
    for i in range(len(docs)):
        docs[i] = os.path.join(os.path.abspath(os.path.dirname(__file__)),docs[i])

    chunk_method='character_recursive'
    chunk_size=400
    chunk_overlap=0
    batch_size=50   # Reduced batch size drastically to not have it be a variable in the test process.
    test_prompt='What are some nuances associated with the analysis and design of hinged booms?'   # Info on test2.pdf

    # Variable inputs
    llm={'OpenAI':ChatOpenAI(model_name='gpt-3.5-turbo-1106', # Openai
                                openai_api_key=OPENAI_API_KEY,
                                max_tokens=500), 
            'Hugging Face':ChatOpenAI(base_url='https://api-inference.huggingface.co/v1',  # Hugging face
                                    model='mistralai/Mistral-7B-Instruct-v0.2',
                                    api_key=HUGGINGFACEHUB_API_TOKEN,
                                    max_tokens=500)}
    query_model={'OpenAI':OpenAIEmbeddings(model='text-embedding-ada-002',openai_api_key=OPENAI_API_KEY),
                    'Voyage':VoyageAIEmbeddings(model='voyage-2',voyage_api_key=VOYAGE_API_KEY),
                    'RAGatouille':'colbert-ir/colbertv2.0'}
    index_type = {index: index for index in ['ChromaDB', 'Pinecone', 'RAGatouille']}
    rag_type = {rag: rag for rag in ['Standard','Parent-Child','Summary']}
    
    setup={
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'VOYAGE_API_KEY': VOYAGE_API_KEY,
        'HUGGINGFACEHUB_API_TOKEN': HUGGINGFACEHUB_API_TOKEN,
        'PINECONE_API_KEY': PINECONE_API_KEY,
        'LOCAL_DB_PATH': LOCAL_DB_PATH,
        'docs': docs,
        'chunk_method': chunk_method,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'batch_size': batch_size,
        'test_prompt': test_prompt,
        'llm': llm,
        'query_model': query_model,
        'index_type': index_type,
        'rag_type': rag_type
    }

    return setup
@pytest.fixture()
def temp_dotenv():
    dotenv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..','.env')
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            print('Creating .env file for testing.')
            f.write(f'OPENAI_API_KEY = {setup_fixture["OPENAI_API_KEY"]}\n')
            f.write(f'PINECONE_API_KEY = {setup_fixture["PINECONE_API_KEY"]}\n')
            f.write(f'HUGGINGFACEHUB_API_TOKEN = {setup_fixture["HUGGINGFACEHUB_API_TOKEN"]}\n')
            f.write(f'LOCAL_DB_PATH = {setup_fixture["LOCAL_DB_PATH"]}\n')
        yield dotenv_path
        os.remove(dotenv_path)
    else:
        yield dotenv_path

### Begin tests
# Test chunk docs
def test_chunk_docs_standard():
    """
    Test the chunk_docs function with standard RAG.

    Args:
        setup_fixture (dict): The setup variables and configurations.
    """
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Standard'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'])
    assert result['rag'] == setup_fixture['rag_type']['Standard']
    assert result['pages'] is not None
    assert result['chunks'] is not None
    assert result['splitters'] is not None
def test_chunk_docs_parent_child():
    """
    Test the chunk_docs function with parent-child RAG.

    Args:
        setup_fixture (dict): The setup variables and configurations.
    """
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Parent-Child'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'])
    assert result['rag'] == setup_fixture['rag_type']['Parent-Child']
    assert result['pages']['doc_ids'] is not None
    assert result['pages']['parent_chunks'] is not None
    assert result['chunks'] is not None
    assert result['splitters']['parent_splitter'] is not None
    assert result['splitters']['child_splitter'] is not None
def test_chunk_docs_summary():
    """
    Test the chunk_docs function with summary RAG.

    Args:
        setup_fixture (dict): The setup variables and configurations.
    """
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Summary'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'], 
                        llm=setup_fixture['llm']['Hugging Face'])
    assert result['rag'] == setup_fixture['rag_type']['Summary']
    assert result['pages']['doc_ids'] is not None
    assert result['pages']['docs'] is not None
    assert result['summaries'] is not None
    assert result['llm'] == setup_fixture['llm']['Hugging Face']

# Test initialize database with a test query
def test_initialize_database_pinecone(monkeypatch):
    """
    Test the initialization of a Pinecone database.

    Args:
        monkeypatch: The monkeypatch fixture.
        setup_fixture (dict): The setup fixture containing the necessary parameters.

    Returns:
        PineconeVectorStore: The initialized Pinecone vector store.

    Raises:
        AssertionError: If the initialized vector store is not an instance of PineconeVectorStore.
    """
    index_type = 'Pinecone'
    index_name = 'test-index'
    query_model = setup_fixture['query_model']['OpenAI']
    rag_type = 'Standard'
    clear = True
    init_ragatouille = False
    show_progress = False

    # Test with environment variable local_db_path
    try:
        vectorstore = initialize_database(index_type, index_name, query_model, rag_type, os.environ['LOCAL_DB_PATH'], clear, init_ragatouille, show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    assert isinstance(vectorstore, PineconeVectorStore)
    delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    # Test with local_db_path set manually, show it doesn't work if not set
    monkeypatch.delenv('LOCAL_DB_PATH', raising=False)
    with pytest.raises(Exception):
        initialize_database(index_type, index_name, query_model, rag_type, os.environ['LOCAL_DB_PATH'], clear, init_ragatouille, show_progress)
    try:    # Probably redudnant but to avoid cleanup
        delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
    except:
        pass
def test_initialize_database_chromadb(monkeypatch):
    """
    Test the initialization of a Chroma database.

    Args:
        monkeypatch: The monkeypatch fixture.
        setup_fixture (dict): A dictionary containing setup fixtures.

    Returns:
        None

    Raises:
        AssertionError: If the vectorstore is not an instance of Chroma.

    """
    index_type = 'ChromaDB'
    index_name = 'test-index'
    query_model = setup_fixture['query_model']['OpenAI']
    rag_type = 'Standard'
    clear = True
    init_ragatouille = False
    show_progress = False

    # Test with environment variable local_db_path
    try:
        vectorstore = initialize_database(index_type, index_name, query_model, rag_type, os.environ['LOCAL_DB_PATH'], clear, init_ragatouille, show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    assert isinstance(vectorstore, Chroma)
    delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    # Test with local_db_path set manually, show it doesn't work if not set
    monkeypatch.delenv('LOCAL_DB_PATH', raising=False)
    with pytest.raises(Exception):
        initialize_database(index_type, index_name, query_model, rag_type, os.environ['LOCAL_DB_PATH'], clear, init_ragatouille, show_progress)
    try:    # Probably redudnant but to avoid cleanup
        delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
    except:
        pass
def test_initialize_database_ragatouille(monkeypatch):
    """
    Test the initialization of a database for RAGatouille.

    Args:
        monkeypatch: The monkeypatch fixture.
        setup_fixture (dict): A dictionary containing the setup fixture.

    Returns:
        None

    Raises:
        AssertionError: If the vectorstore is not an instance of RAGPretrainedModel.

    """
    index_type = 'RAGatouille'
    index_name = 'test-index'
    query_model = setup_fixture['query_model']['RAGatouille']
    rag_type = 'Standard'
    local_db_path = setup_fixture['LOCAL_DB_PATH']
    clear = True
    init_ragatouille = True
    show_progress = False

    # Test with environment variable local_db_path
    try:
        vectorstore = initialize_database(index_type, index_name, query_model, rag_type, os.environ['LOCAL_DB_PATH'], clear, init_ragatouille, show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
    
    assert isinstance(vectorstore, RAGPretrainedModel)
    delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    # Test with local_db_path set manually, show it doesn't work if not set
    monkeypatch.delenv('LOCAL_DB_PATH', raising=False)
    with pytest.raises(Exception):
        initialize_database(index_type, index_name, query_model, rag_type, os.environ['LOCAL_DB_PATH'], clear, init_ragatouille, show_progress)
    try:    # Probably redudnant but to avoid cleanup
        delete_index(index_type, index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
    except:
        pass

# Test end to end process, adding query
def test_database_setup_and_query(test_input):
    """Tests the entire process of initializing a database, upserting documents, and deleting a database.

    Args:
        setup_fixture (dict): The setup fixture containing the necessary configuration for the test.
        test_input (str): The test input.

    Raises:
        Exception: If there is an error during the test.

    Returns:
        None
    """
    test, print_str = parse_test_case(setup_fixture,test_input)
    index_name='test'+str(test['id'])
    print(f"Starting test: {print_str}")

    try:            
        vectorstore = load_docs(
            test['index_type'],
            setup_fixture['docs'],
            rag_type=test['rag_type'],
            query_model=test['query_model'],
            index_name=index_name, 
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            clear=True,
            batch_size=setup_fixture['batch_size'],
            local_db_path=setup_fixture['LOCAL_DB_PATH'],
            llm=test['llm'])
        if test['index_type'] == 'ChromaDB':
            assert isinstance(vectorstore, Chroma)
        elif test['index_type'] == 'Pinecone':
            assert isinstance(vectorstore, PineconeVectorStore)
        elif test['index_type'] == 'RAGatouille':
            assert isinstance(vectorstore, RAGPretrainedModel)
        print('Vectorstore created.')

        # Set index names for special databases
        if test['rag_type'] == 'Parent-Child':
            index_name = index_name + '-parent-child'
        if test['rag_type'] == 'Summary':
            index_name = index_name + '-summary-' + test['llm'].model_name.replace('/', '-')

        if test['index_type'] == 'RAGatouille':
            query_model_qa = RAGPretrainedModel.from_index(
                os.path.join(setup_fixture['LOCAL_DB_PATH'],'.ragatouille/colbert/indexes','test'+str(test['id'])),
                n_gpu=0,
                verbose=0)             
        else:
            query_model_qa = test['query_model']
        assert query_model_qa is not None
        
        qa_model_obj = QA_Model(test['index_type'],
                            index_name,
                            query_model_qa,
                            test['llm'],
                            rag_type=test['rag_type'],
                            local_db_path=setup_fixture['LOCAL_DB_PATH'])
        print('QA model object created.')
        assert qa_model_obj is not None

        qa_model_obj.query_docs(setup_fixture['test_prompt'])
        response = qa_model_obj.ai_response
        assert response is not None

        alternate_question = qa_model_obj.generate_alternative_questions(setup_fixture['test_prompt'], response=response)
        assert alternate_question is not None
        print('Query and alternative question successful!')

        delete_index(test['index_type'],
                index_name, 
                test['rag_type'],
                local_db_path=setup_fixture['LOCAL_DB_PATH'])
        if test['rag_type'] == 'Parent-Child' or test['rag_type'] == 'Summary':
            lfs_path = os.path.join(setup_fixture['LOCAL_DB_PATH'], 'local_file_Store', index_name)
            assert not os.path.exists(lfs_path) # Check that the local file store was deleted
        print('Database deleted.')

    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test['index_type'],
                        'test'+str(test['id']), 
                        test['rag_type'],
                        local_db_path=setup_fixture['LOCAL_DB_PATH'])
        raise e
        
# Test sidebar loading and secret keys
def test_load_sidebar():
    '''
    This function tests the functionality of the load_sidebar function by passing different combinations of arguments and checking the returned sidebar configuration.

    Returns:
        None
    '''
    # TODO Add mock changes from streamlit changing: index_type, rag_type

    # Use the existing config files, to check they are set up correctly.
    base_folder_path = os.path.abspath(os.path.dirname(__file__))
    base_folder_path = os.path.join(base_folder_path, '..')
    base_folder_path = os.path.normpath(base_folder_path)
    config_file=os.path.join(base_folder_path, 'config', 'config.json')
    index_data_file=os.path.join(base_folder_path, 'config', 'index_data.json')

    # Test case: Only embeddings is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, vector_database=True, embeddings=True)
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'Openai'

    # Test case: Only rag_type is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, vector_database=True, rag_type=True)
    assert 'rag_type' in sidebar_config
    assert sidebar_config['rag_type'] == 'Standard'

    # Test case: Only index_name is True (should give valuerror)
    with pytest.raises(ValueError):
        sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, vector_database=True, index_name=True)    

    # Test case: Only embeddings, index_name and rag_type are True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, vector_database=True, embeddings=True, index_name=True, rag_type=True)
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'Openai'
    assert 'index_name' in sidebar_config
    assert sidebar_config['index_name'] == 'chromadb-openai'

    # Test case: Only llm is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, vector_database=True, llm=True)
    assert 'llm_source' in sidebar_config
    assert sidebar_config['llm_source'] == 'OpenAI'

    # Test case: Only model_options is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, vector_database=True, model_options=True)
    assert 'temperature' in sidebar_config['model_options']
    assert sidebar_config['model_options']['temperature'] == 0.1
    assert 'output_level' in sidebar_config['model_options']
    assert sidebar_config['model_options']['output_level'] == 1000

    # Test case: All options are True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, vector_database=True,
                                  embeddings=True, rag_type=True, index_name=True, llm=True, model_options=True)
    assert 'index_type' in sidebar_config
    assert sidebar_config['index_type'] == 'ChromaDB'
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'Openai'
    assert 'rag_type' in sidebar_config
    assert sidebar_config['rag_type'] == 'Standard'
    assert 'index_name' in sidebar_config
    assert sidebar_config['index_name'] == 'chromadb-openai'
    assert 'llm_source' in sidebar_config
    assert sidebar_config['llm_source'] == 'OpenAI'
    assert 'temperature' in sidebar_config['model_options']
    assert sidebar_config['model_options']['temperature'] == 0.1
    assert 'output_level' in sidebar_config['model_options']
    assert sidebar_config['model_options']['output_level'] == 1000
def test_set_secrets_with_environment_variables(monkeypatch):
    """
    Test case to verify the behavior of the set_secrets function when environment variables are set.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables during testing.

    Returns:
        None

    Raises:
        AssertionError: If the secrets are not set correctly.
    """
    # Set the environment variables
    monkeypatch.setenv('OPENAI_API_KEY', 'openai_key')
    monkeypatch.setenv('VOYAGE_API_KEY', 'voyage_key')
    monkeypatch.setenv('PINECONE_API_KEY', 'pinecone_key')
    monkeypatch.setenv('HUGGINGFACEHUB_API_TOKEN', 'huggingface_key')
    # Call the set_secrets function
    secrets = set_secrets({})
    # Assert that the secrets are set correctly
    assert secrets['OPENAI_API_KEY'] == 'openai_key'
    assert secrets['VOYAGE_API_KEY'] == 'voyage_key'
    assert secrets['PINECONE_API_KEY'] == 'pinecone_key'
    assert secrets['HUGGINGFACEHUB_API_TOKEN'] == 'huggingface_key'
def test_set_secrets_with_inputs(monkeypatch):
    """
    Test case for the set_secrets function with sidebar data.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables.

    Returns:
        None

    Raises:
        AssertionError: If the secrets are not set correctly.
    """
    # For this test, delete the environment variables
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('VOYAGE_API_KEY', raising=False)
    monkeypatch.delenv('PINECONE_API_KEY', raising=False)
    monkeypatch.delenv('HUGGINGFACEHUB_API_TOKEN', raising=False)
    # Define the sidebar data
    sb = {
        'keys': {
            'OPENAI_API_KEY': 'openai_key',
            'VOYAGE_API_KEY': 'voyage_key',
            'PINECONE_API_KEY': 'pinecone_key',
            'HUGGINGFACEHUB_API_TOKEN': 'huggingface_key'
        }
    }
    # Call the set_secrets function
    secrets = set_secrets(sb)
    # Assert that the secrets are set correctly
    assert secrets['OPENAI_API_KEY'] == 'openai_key'
    assert secrets['VOYAGE_API_KEY'] == 'voyage_key'
    assert secrets['PINECONE_API_KEY'] == 'pinecone_key'
    assert secrets['HUGGINGFACEHUB_API_TOKEN'] == 'huggingface_key'
@pytest.mark.parametrize("missing_key",
                         ['OPENAI_API_KEY','VOYAGE_API_KEY','PINECONE_API_KEY','HUGGINGFACEHUB_API_TOKEN'])
def test_set_secrets_missing_api_keys(monkeypatch, missing_key):
    """
    Test case for setting secrets with missing API keys.

    Args:
        monkeypatch: A pytest fixture for patching values during testing.
        missing_key: The missing API key to be tested.

    Raises:
        SecretKeyException: If the set_secrets function raises an exception.

    Returns:
        None
    """
    print(f"Testing missing required key: {missing_key}")
    # For this test, delete the environment variables
    key_list=['OPENAI_API_KEY','VOYAGE_API_KEY','PINECONE_API_KEY','HUGGINGFACEHUB_API_TOKEN']
    for key in key_list:
        monkeypatch.delenv(key, raising=False)
    # Define the sidebar data with the current key being tested set to an empty string
    sb = {'keys': {missing_key: ''}}
    # Call the set_secrets function without setting any environment variables or sidebar data
    with pytest.raises(SecretKeyException):
        set_secrets(sb)

# Test streamlit setup
def test_st_setup_page_local_db_path_only_defined(monkeypatch):
    """
    Test case for the `st_setup_page` function when only the local db path is defined.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables during testing.

    Returns:
        None
    """
    page_title = "Test Page"

    # Clear all environment variables
    for var in list(os.environ.keys()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr('admin.load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setattr('admin.find_dotenv', lambda *args, **kwargs: '')  # Assuming an empty string simulates not finding a .env file
    monkeypatch.setenv('LOCAL_DB_PATH', os.path.abspath(os.path.dirname(__file__))) # Set the local db path to this directory

    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)

    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir)

    # Assert
    assert paths['db_folder_path'] == os.getenv('LOCAL_DB_PATH')
    assert sb == {}
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': None,
                       'OPENAI_API_KEY': None,
                       'PINECONE_API_KEY': None,
                       'VOYAGE_API_KEY': None}
def test_st_setup_page_local_db_path_not_defined(monkeypatch):
    """
    Test case to verify the behavior of st_setup_page function when LOCAL_DB_PATH is not defined.

    Args:
        monkeypatch: Monkeypatch object for modifying environment variables.

    Returns:
        None
    """

    page_title = "Test Page"

    # Clear all environment variables
    for var in list(os.environ.keys()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr('admin.load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setattr('admin.find_dotenv', lambda *args, **kwargs: '')  # Assuming an empty string simulates not finding a .env file
    monkeypatch.setenv('LOCAL_DB_PATH', None) # Set to none

    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)

    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir)

    # Assert
    assert paths['db_folder_path'] == 'None'
    assert sb == {}
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': None,
                       'OPENAI_API_KEY': None,
                       'PINECONE_API_KEY': None,
                       'VOYAGE_API_KEY': None}
def test_st_setup_page_local_db_path_w_all_man_input(monkeypatch):
    """
    Test case for the st_setup_page function with all inputs in sidebar and manual input for environment variables.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables and other attributes during testing.

    Returns:
        None
    """

    page_title = "Test Page"
    sidebar_config = {
        'vector_database': True,
        'embeddings': True,
        'rag_type': True,
        'index_name': True,
        'llm': True,
        'model_options': True
    }

    # Set all environment variables, simulate manual input no .env
    monkeypatch.setenv('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
    monkeypatch.setenv('VOYAGE_API_KEY', os.getenv('VOYAGE_API_KEY'))
    monkeypatch.setenv('PINECONE_API_KEY', os.getenv('PINECONE_API_KEY'))
    monkeypatch.setenv('HUGGINGFACEHUB_API_TOKEN', os.getenv('HUGGINGFACEHUB_API_TOKEN'))
    monkeypatch.setattr('admin.load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setattr('admin.find_dotenv', lambda *args, **kwargs: '')  # Assuming an empty string simulates not finding a .env file
    monkeypatch.setenv('LOCAL_DB_PATH', os.path.abspath(os.path.dirname(__file__))) # Set the local db path to this directory

    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)

    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir, sidebar_config)

    # Assert
    assert paths['db_folder_path'] == os.getenv('LOCAL_DB_PATH')
    assert isinstance(sb, dict) and sb != {}    # Test that it's not an empty dictionary
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                       'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                       'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
                       'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY')}
def test_st_setup_page_local_db_path_w_all_env_input(monkeypatch,temp_dotenv):
    """
    Test case for the `st_setup_page` function with all inputs in sidebar and all environment variables set using .env file.

    Args:
        monkeypatch: Monkeypatch object for modifying environment variables.

    Returns:
        None
    """

    page_title = "Test Page"
    sidebar_config = {
        'vector_database': True,
        'embeddings': True,
        'rag_type': True,
        'index_name': True,
        'llm': True,
        'model_options': True
    }

    # Clear all environment variables, simulate .env load in st_setup_page and pre-set local_db_path
    for var in list(os.environ.keys()):
        monkeypatch.delenv(var, raising=False)
    dotenv_path = temp_dotenv
    print(dotenv_path)

    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)

    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir, sidebar_config)

    # Assert
    assert paths['db_folder_path'] == os.getenv('LOCAL_DB_PATH')
    assert isinstance(sb, dict) and sb != {}    # Test that it's not an empty dictionary
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                       'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                       'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
                       'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY')}

# Test data visualization
def test_reduce_vector_query_size():
    """
    Test function to verify the behavior of the reduce_vector_query_size function.

    Args:
        setup_fixture: The setup fixture for the test.

    Raises:
        Exception: If an error occurs during the test.

    Returns:
        None
    """
    index_name = 'test-index'
    rx_client, chroma_client = viz_database_setup(index_name, setup_fixture)
    try:
        collection = chroma_client.get_collection(name=index_name, embedding_function=rx_client._chosen_embedding_model)
        rx_client.load_chroma(collection, initialize_projector=True)
        vector_qty = 3  # Do a small quantity for the test

        rx_client = reduce_vector_query_size(rx_client, chroma_client, vector_qty, verbose=True)
        assert len(rx_client._documents.embeddings) == vector_qty
        assert len(rx_client._documents.text) == vector_qty
        assert len(rx_client._documents.ids) == vector_qty
        chroma_client.delete_collection(name=rx_client._vectordb.name)
    except Exception as e:
        chroma_client.delete_collection(name=rx_client._vectordb.name)
        raise e
def test_create_data_viz_no_limit():
    """
    Test case: Without limit_size_qty and df_export_path

    This test case verifies the behavior of the create_data_viz function when called without providing
    the limit_size_qty and df_export_path parameters. It performs the following steps:
    1. Sets up the necessary fixtures for the test.
    2. Sets up the RX and Chroma clients for visualization database.
    3. Calls the create_data_viz function with the index_name, rx_client, and chroma_client.
    4. Verifies that the returned rx_client_out is an instance of RAGxplorer.
    5. Verifies that the returned chroma_client_out is an instance of ClientAPI.
    6. Verifies that the name of the RX client's vector database contains the index_name.
    7. Deletes the collection associated with the RX client's vector database.

    Raises:
        Any exception raised during the test.

    """
    index_name = 'test-index'
    rx_client, chroma_client = viz_database_setup(index_name,setup_fixture)
    try:
        rx_client_out, chroma_client_out = create_data_viz(index_name, rx_client, chroma_client)
    except Exception as e:
        try:
            chroma_client.delete_collection(name=rx_client._vectordb.name)
        except:
            pass
        raise e   
    assert isinstance(rx_client_out, RAGxplorer)
    assert isinstance(chroma_client_out, ClientAPI)
    assert "test-index" in rx_client_out._vectordb.name
    chroma_client.delete_collection(name=rx_client_out._vectordb.name)
def test_create_data_viz_limit():
    '''
    Test case: With limit_size_qty and df_export_path

    This test case verifies the behavior of the create_data_viz function when called without providing
    the limit_size_qty and df_export_path parameters. It performs the following steps:
    1. Sets up the necessary fixtures for the test.
    2. Sets up the RX and Chroma clients for visualization database.
    3. Calls the create_data_viz function with the index_name, rx_client, and chroma_client.
    4. Verifies that the returned rx_client_out is an instance of RAGxplorer.
    5. Verifies that the returned chroma_client_out is an instance of ClientAPI.
    6. Verifies that the name of the RX client's vector database contains the index_name.
    7. Deletes the collection associated with the RX client's vector database.

    Raises:
        Any exception raised during the test.

    '''
    index_name = 'test-index'
    export_file='data_viz_test.json'
    rx_client, chroma_client = viz_database_setup(index_name,setup_fixture)
    try:
        rx_client_out, chroma_client_out = create_data_viz(
            index_name, rx_client, chroma_client, limit_size_qty=10, df_export_path=os.path.join(setup_fixture['LOCAL_DB_PATH'],export_file))
    except Exception as e:
        try:
            chroma_client.delete_collection(name=rx_client._vectordb.name)
        except:
            pass
        try:
            os.remove(os.path.join(setup_fixture['LOCAL_DB_PATH'], export_file))
        except:
            pass
        raise e   
    assert isinstance(rx_client_out, RAGxplorer)
    assert isinstance(chroma_client_out, ClientAPI)
    assert "test-index" in rx_client_out._vectordb.name
    assert os.path.exists(os.path.join(setup_fixture['LOCAL_DB_PATH'], export_file))
    chroma_client.delete_collection(name=rx_client_out._vectordb.name)
    os.remove(os.path.join(setup_fixture['LOCAL_DB_PATH'], export_file))