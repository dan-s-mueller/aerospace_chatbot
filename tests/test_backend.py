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

# Import local variables
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src/aerospace_chatbot'))

from data_processing import chunk_docs, initialize_database, load_docs, delete_index 
from admin import _get_base_path, load_sidebar, set_secrets, show_pinecone_indexes, SecretKeyException
from queries import QA_Model

def permute_tests(test_data):
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
    # This script should be run first to generate a json which is read and dynamically sets test cases when running pytest.
    # Determine the set of cases to screen

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
    Read json data from generate_test_cases
    '''
    with open(json_path, 'r') as json_file:
        test_cases = json.load(json_file)
    return test_cases
def pytest_generate_tests(metafunc):
    '''
    Use pytest_generate_tests to dynamically generate tests
    Tests generates tests from a static file (test_cases.json). You must run generate_test_cases() first.
    '''
    if "test_input" in metafunc.fixturenames:
        tests = read_test_cases(os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_cases.json'))
        metafunc.parametrize("test_input", tests)
def parse_test_case(setup,test_case):
    ''' 
    Parse test case to be used in the test functions.
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

@pytest.fixture(scope="session", autouse=True)
def setup_fixture():
    """
    Sets up the necessary variables and configurations for the test.

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

### Begin tests
# Test chunk docs
def test_chunk_docs_standard(setup_fixture):
    # Test case for chunk docs: Standard rag
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Standard'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'])
    assert result['rag'] == setup_fixture['rag_type']['Standard']
    assert result['pages'] is not None
    assert result['chunks'] is not None
    assert result['splitters'] is not None
def test_chunk_docs_parent_child(setup_fixture):
    # Test case for chunk docs: Parent-Child rag
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
def test_chunk_docs_summary(setup_fixture):
    # Test case for chunk docs: Summary rag
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
def test_initialize_database_pinecone(setup_fixture):
    index_type = 'Pinecone'
    index_name = 'test-index'
    query_model = setup_fixture['query_model']['OpenAI']
    rag_type = 'Standard'
    local_db_path = setup_fixture['LOCAL_DB_PATH']
    clear = True
    init_ragatouille = False
    show_progress = False

    vectorstore = initialize_database(index_type, index_name, query_model, rag_type, local_db_path, clear, init_ragatouille, show_progress)
    assert isinstance(vectorstore, PineconeVectorStore)
    delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
def test_initialize_database_chromadb(setup_fixture):
    index_type = 'ChromaDB'
    index_name = 'test-index'
    query_model = setup_fixture['query_model']['OpenAI']
    rag_type = 'Standard'
    local_db_path = setup_fixture['LOCAL_DB_PATH']
    clear = True
    init_ragatouille = False
    show_progress = False

    try:
        vectorstore = initialize_database(index_type, index_name, query_model, rag_type, local_db_path, clear, init_ragatouille, show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)

    assert isinstance(vectorstore, Chroma)
    delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
def test_initialize_database_ragatouille(setup_fixture):
    index_type = 'RAGatouille'
    index_name = 'test-index'
    query_model = setup_fixture['query_model']['RAGatouille']
    rag_type = 'Standard'
    local_db_path = setup_fixture['LOCAL_DB_PATH']
    clear = True
    init_ragatouille = True
    show_progress = False

    try:
        vectorstore = initialize_database(index_type, index_name, query_model, rag_type, local_db_path, clear, init_ragatouille, show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
    
    assert isinstance(vectorstore, RAGPretrainedModel)
    delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)

# Test end to end process, adding query
def test_database_setup_and_query(setup_fixture,test_input):
    """  
    Tests the entire process of initializing a database, upserting documents, and deleting a database.
    Test cases combining Pinecone and Voyage will often appear to timeout. Try changing the API key for Voyage, this seems to fix the issue. It's not a problem with Pinecone.
    """
    test, print_str =parse_test_case(setup_fixture,test_input)

    print(f"Starting test: {print_str}")

    try:            
        vectorstore = load_docs(
            test['index_type'],
            setup_fixture['docs'],
            rag_type=test['rag_type'],
            query_model=test['query_model'],
            index_name='test'+str(test['id']), 
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

        if test['index_type'] == 'RAGatouille':
            query_model_qa = RAGPretrainedModel.from_index(
                os.path.join(setup_fixture['LOCAL_DB_PATH'],'.ragatouille/colbert/indexes','test'+str(test['id'])),
                n_gpu=0,
                verbose=0)             
        else:
            query_model_qa = test['query_model']
        assert query_model_qa is not None
        
        qa_model_obj = QA_Model(test['index_type'],
                            'test'+str(test['id']),
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
                'test'+str(test['id']), 
                test['rag_type'],
                local_db_path=setup_fixture['LOCAL_DB_PATH'])
        print('Database deleted.')

    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test['index_type'],
                        'test'+str(test['id']), 
                        test['rag_type'],
                        local_db_path=setup_fixture['LOCAL_DB_PATH'])
        raise e
        
# TODO add tests for path setup. test config, data, db paths
def test_load_sidebar():
    '''
    Test load_sidebar function.
    '''
    # TODO Add mock changes from streamlit changing: index_type, rag_type

    # Use the existing config files, to check they are set up correctly.
    base_folder_path = os.path.abspath(os.path.dirname(__file__))
    base_folder_path = os.path.join(base_folder_path, '..')
    base_folder_path = os.path.normpath(base_folder_path)
    config_file=os.path.join(base_folder_path, 'config', 'config.json')
    index_data_file=os.path.join(base_folder_path, 'config', 'index_data.json')

    # Test case: Only embeddings is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, embeddings=True)
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'Openai'

    # Test case: Only rag_type is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, rag_type=True)
    assert 'rag_type' in sidebar_config
    assert sidebar_config['rag_type'] == 'Standard'

    # Test case: Only index_name is True (should give valuerror)
    with pytest.raises(ValueError):
        sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, index_name=True)    

    # Test case: Only embeddings and index_name are True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, embeddings=True, index_name=True)
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'Openai'
    assert 'index_name' in sidebar_config
    assert sidebar_config['index_name'] == 'chromadb-openai'

    # Test case: Only llm is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, llm=True)
    assert 'llm_source' in sidebar_config
    assert sidebar_config['llm_source'] == 'OpenAI'

    # Test case: Only model_options is True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file, model_options=True)
    assert 'temperature' in sidebar_config['model_options']
    assert sidebar_config['model_options']['temperature'] == 0.1
    assert 'output_level' in sidebar_config['model_options']
    assert sidebar_config['model_options']['output_level'] == 1000

    # Test case: All options are True
    sidebar_config = load_sidebar(config_file=config_file, index_data_file=index_data_file,
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

# Test secret key setup
def test_env_variables_exist(setup_fixture):
    assert setup_fixture['OPENAI_API_KEY'] is not None
    assert setup_fixture['VOYAGE_API_KEY'] is not None
    assert setup_fixture['HUGGINGFACEHUB_API_TOKEN'] is not None
    assert setup_fixture['PINECONE_API_KEY'] is not None
    print('Environment variables test passed.')
def test_set_secrets_with_environment_variables(monkeypatch):
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
def test_set_secrets_with_sidebar_data(monkeypatch):
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