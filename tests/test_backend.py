import os, sys, json
import itertools
import pytest
import pandas as pd
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from ragatouille import RAGPretrainedModel

import chromadb
from chromadb import ClientAPI

# Import local variables
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src/aerospace_chatbot'))
from data_processing import chunk_docs, initialize_database, load_docs, \
      delete_index, _stable_hash_meta, get_docs_df, get_questions_df, \
      add_clusters, get_docs_questions_df
from admin import load_sidebar, set_secrets, st_setup_page, SecretKeyException
from queries import QA_Model

# TODO test eval.py functionality
# TODO add tests to check conversation history functionality
# TODO add mock changes from streamlit changing: index_type, rag_type. or just test streamlit frontend.
# TODO test db_name functionv

# Functions
def permute_tests(test_data):
    '''
    Generate permutations of test cases.

    Args:
        test_data (list): List of dictionaries containing test case data.

    Returns:
        list: List of dictionaries representing permuted test cases.
    '''
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
def generate_test_cases(config_file:str,export:bool=True,export_dir:str='.'):
    """
    Generates test cases based on the provided configuration file. Exports a snapshot of this which is used in the tests.
    Args:
        config_file (str): The path to the configuration file.
        export (bool, optional): Whether to export the generated test cases to a file. Defaults to True.
        export_dir (str, optional): The directory to export the test cases file to. Defaults to '.'.
    Returns:
        List[Dict[str, Union[str, List[str]]]]: A list of test cases, where each test case is a dictionary with the following keys:
            - 'index_type' (str): The type of index.
            - 'query_model' (str): The query model.
            - 'embedding_name' (List[str]): The names of the embedding models.
            - 'rag_type' (str): The type of RAG.
            - 'llm_family' (str): The family of language model.
            - 'llm' (List[str]): The names of the language models.
    Raises:
        FileNotFoundError: If the provided configuration file does not exist.
    """
    # Items in test_cases must match labels to select from in setup_fixture
    # TODO throw in bad inputs for config, embeddings_list, llms below

    with open(config_file, 'r') as f:
        config = json.load(f)
        embeddings_list = {e['name']: e for e in config['embeddings']}
        llms  = {m['name']: m for m in config['llms']}

    test_cases = [
        {
            # Tests ChromaDB setups, standard RAG , openai embeddings
            'index_type' : ['ChromaDB'],
            'query_model' : ['OpenAI'],
            'embedding_name' : embeddings_list['OpenAI']['embedding_models'],
            'rag_type' : ['Standard'],
            'llm_family' : ['OpenAI'],
            'llm' : [llms['OpenAI']['models'][0]]
        },
        {
            # Tests standard RAG , openai llms
            'index_type' : ['ChromaDB'],
            'query_model' : ['OpenAI'],
            'embedding_name' : [embeddings_list['OpenAI']['embedding_models'][0]],
            'rag_type' : ['Standard'],
            'llm_family' : ['OpenAI'],
            'llm' : llms['OpenAI']['models']
        },
        {
            # Tests standard RAG , hugging face embeddings
            'index_type' : ['ChromaDB'],
            'query_model' : ['Hugging Face'],
            'embedding_name' : embeddings_list['Hugging Face']['embedding_models'],
            'rag_type' : ['Standard'],
            'llm_family' : ['OpenAI'],
            'llm' : [llms['OpenAI']['models'][0]]
        },
        {
            # Tests standard RAG , all hugging face llms
            'index_type' : ['ChromaDB'],
            'query_model' : ['OpenAI'],
            'embedding_name' : [embeddings_list['OpenAI']['embedding_models'][0]],
            'rag_type' : ['Standard'],
            'llm_family' : ['Hugging Face'],
            'llm' : llms['Hugging Face']['models']
        },
        {
            # Tests parent-child rag, openai models
            'index_type' : ['ChromaDB'],
            'query_model' : ['OpenAI'],
            'embedding_name' : [embeddings_list['OpenAI']['embedding_models'][0]],
            'rag_type' : ['Parent-Child'],
            'llm_family' : ['OpenAI'],
            'llm' : [llms['OpenAI']['models'][0]]
        },
        {
            # Tests advanced RAG (summary), openai models
            'index_type' : ['ChromaDB'],
            'query_model' : ['OpenAI'],
            'embedding_name' : [embeddings_list['OpenAI']['embedding_models'][0]],
            'rag_type' : ['Summary'],
            'llm_family' : ['OpenAI'],
            'llm' : [llms['OpenAI']['models'][0]]
        },
        {
            # Tests Pinecone setups, openai embedding type
            'index_type' : ['Pinecone'],
            'query_model' : ['OpenAI'],
            'embedding_name' : [embeddings_list['OpenAI']['embedding_models'][0]],
            'rag_type' : ['Standard'],
            'llm_family' : ['OpenAI'],
            'llm' : [llms['OpenAI']['models'][0]]
        },
        {
            # Tests Pinecone setups, voyage embedding types
            'index_type' : ['Pinecone'],
            'query_model' : ['Voyage'],
            'embedding_name' : embeddings_list['Voyage']['embedding_models'],
            'rag_type' : ['Standard'],
            'llm_family' : ['OpenAI'],
            'llm' : [llms['OpenAI']['models'][0]]
        },
        {
            # Tests RAGatouille setup
            'index_type' : ['RAGatouille'],
            'query_model' : ['RAGatouille'],
            'embedding_name' : ['colbert-ir/colbertv2.0'],  # Hard coded, tied to RAGatouille
            'rag_type' : ['Standard'],
            'llm_family' : ['OpenAI'],
            'llm' : [llms['OpenAI']['models'][0]]
        }
    ]

    tests = permute_tests(test_cases)
    
    if export:
        file_path = os.path.join(export_dir,'test_cases.json')
        with open(file_path, 'w') as json_file:
            json.dump(tests, json_file, indent=4)
    
    return tests
def read_test_cases(json_path:str):
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
    if 'test_input' in metafunc.fixturenames:
        tests = read_test_cases(os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_cases.json'))
        metafunc.parametrize('test_input', tests)
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
        'query_model': test_case['query_model'],
        'embedding_name': test_case['embedding_name'],
        'rag_type': setup['rag_type'][test_case['rag_type']],
        'llm_family': test_case['llm_family'],
        'llm' : test_case['llm']
    }
    print_str = ', '.join(f'{key}: {value}' for key, value in test_case.items())

    return parsed_test, print_str
def parse_test_model(type, test, setup_fixture):
    """
    Parses the test model based on the given type and test parameters.

    Args:
        type (str): The type of the test model ('embedding' or 'llm').
        test (dict): The test parameters.
        setup_fixture (dict): The setup fixture containing API keys.

    Returns:
        object: The parsed test model.

    Raises:
        NotImplementedError: If the query model or LLM is not implemented.
    """
    if type == 'embedding':
        # Parse out embedding
        if test['index_type'] == 'RAGatouille':
            query_model = RAGPretrainedModel.from_pretrained(test['embedding_name'],
                                                             index_root=os.path.join(setup_fixture['LOCAL_DB_PATH'],'.ragatouille'))
        elif test['query_model'] == 'OpenAI' or test['query_model'] == 'Voyage' or test['query_model'] == 'Hugging Face':
            if test['query_model'] == 'OpenAI':
                query_model = OpenAIEmbeddings(model=test['embedding_name'], openai_api_key=setup_fixture['OPENAI_API_KEY'])
            elif test['query_model'] == 'Voyage':
                query_model = VoyageAIEmbeddings(model=test['embedding_name'], voyage_api_key=setup_fixture['VOYAGE_API_KEY'], truncation=False)
            elif test['query_model'] == 'Hugging Face':
                query_model = HuggingFaceInferenceAPIEmbeddings(model_name=test['embedding_name'], 
                                                                api_key=setup_fixture['HUGGINGFACEHUB_API_TOKEN'])
        else:
            raise NotImplementedError('Query model not implemented.')
        return query_model
    elif type == 'llm':
        # Parse out llm
        if test['llm_family'] == 'OpenAI':
            llm = ChatOpenAI(model_name=test['llm'], openai_api_key=setup_fixture['OPENAI_API_KEY'], max_tokens=500)
        elif test['llm_family'] == 'Hugging Face':
            llm = ChatOpenAI(base_url='https://api-inference.huggingface.co/v1', model=test['llm'], api_key=setup_fixture['HUGGINGFACEHUB_API_TOKEN'], max_tokens=500)
        else:
            raise NotImplementedError('LLM not implemented.')
        return llm
    else:
        raise ValueError('Invalid type. Must be either "embedding" or "llm".')

# Fixtures
@pytest.fixture(scope='session', autouse=True)
def setup_fixture():
    """
    Sets up the fixture for testing the backend.
    Returns:
        dict: A dictionary containing the setup configuration.
    Raises:
        None
    """
    ...
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
    batch_size=50
    test_prompt='What are some nuances associated with the analysis and design of hinged booms?'   # Info on test2.pdf

    # Variable inputs
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
        'index_type': index_type,
        'rag_type': rag_type
    }

    return setup
@pytest.fixture()
def temp_dotenv(setup_fixture):
    """
    Creates a temporary .env file for testing purposes.

    Args:
        setup_fixture (dict): A dictionary containing setup fixture values.

    Yields:
        str: The path of the temporary .env file.

    Raises:
        None

    Returns:
        None
    """
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
def test_chunk_docs_standard(setup_fixture):
    '''
    Test the chunk_docs function with standard RAG.

    Args:
        setup_fixture (dict): The setup variables and configurations.
    '''
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Standard'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'])
    
    page_ids = [_stable_hash_meta(page.metadata) for page in result['pages']]
    chunk_ids = [_stable_hash_meta(chunk.metadata) for chunk in result['chunks']]
    
    assert result['rag'] == setup_fixture['rag_type']['Standard']
    assert result['pages'] is not None
    assert len(page_ids) == len(set(page_ids))
    assert result['chunks'] is not None
    assert len(chunk_ids) == len(set(chunk_ids))
    assert result['splitters'] is not None
def test_chunk_docs_merge_nochunk(setup_fixture):
    """
    Test case for the `chunk_docs` function with no chunking and merging.

    Args:
        setup_fixture (dict): A dictionary containing the setup fixture data.

    Returns:
        None
    """

    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Standard'], 
                        chunk_method='None',
                        n_merge_pages=2)
    
    page_ids = [_stable_hash_meta(page.metadata) for page in result['pages']]
    chunk_ids = [_stable_hash_meta(chunk.metadata) for chunk in result['chunks']]

    assert result['rag'] == setup_fixture['rag_type']['Standard']
    assert result['pages'] is not None
    assert len(page_ids) == len(set(page_ids))
    assert result['chunks'] is not None
    assert chunk_ids==page_ids
    assert result['splitters'] is None
def test_chunk_docs_nochunk(setup_fixture):
    '''
    Test the chunk_docs function with no chunking.

    Args:
        setup_fixture (dict): The setup variables and configurations.
    '''
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Standard'], 
                        chunk_method='None', 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'])
    
    page_ids = [_stable_hash_meta(page.metadata) for page in result['pages']]
    chunk_ids = [_stable_hash_meta(chunk.metadata) for chunk in result['chunks']]

    assert result['rag'] == setup_fixture['rag_type']['Standard']
    assert result['pages'] is not None
    assert len(page_ids) == len(set(page_ids))
    assert result['chunks'] is result['pages']
    assert len(chunk_ids) == len(set(chunk_ids))
    assert result['splitters'] is None
def test_chunk_docs_parent_child(setup_fixture):
    '''
    Test the chunk_docs function with parent-child RAG.

    Args:
        setup_fixture (dict): The setup variables and configurations.
    '''
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Parent-Child'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'])
        
    page_ids = [_stable_hash_meta(page.metadata) for page in result['pages']['parent_chunks']]
    chunk_ids = [_stable_hash_meta(chunk.metadata) for chunk in result['chunks']]

    assert result['rag'] == setup_fixture['rag_type']['Parent-Child']
    assert result['pages']['doc_ids'] is not None
    assert result['pages']['parent_chunks'] is not None
    assert len(page_ids) == len(set(page_ids))
    assert result['chunks'] is not None
    assert len(chunk_ids) == len(set(chunk_ids))
    assert result['splitters']['parent_splitter'] is not None
    assert result['splitters']['child_splitter'] is not None
def test_chunk_docs_summary(setup_fixture):
    '''
    Test the chunk_docs function with summary RAG.

    Args:
        setup_fixture (dict): The setup variables and configurations.
    '''
    llm=parse_test_model('llm', {'llm_family': 'OpenAI', 'llm': 'gpt-3.5-turbo-0125'}, setup_fixture)

    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Summary'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'], 
                        llm=llm)
    
    page_ids = [_stable_hash_meta(page.metadata) for page in result['pages']['docs']]
    summary_ids = [_stable_hash_meta(summary.metadata) for summary in result['summaries']]
    
    assert result['rag'] == setup_fixture['rag_type']['Summary']
    assert result['pages']['doc_ids'] is not None
    assert result['pages']['docs'] is not None
    assert len(page_ids) == len(set(page_ids))
    assert result['summaries'] is not None
    assert len(summary_ids) == len(set(summary_ids))
    assert result['llm'] == llm
def test_chunk_id_lookup(setup_fixture):
    '''
    Test case for chunk_id_lookup function.

    Args:
        setup_fixture (dict): A dictionary containing setup fixtures.

    Returns:
        None
    '''
    result = chunk_docs(setup_fixture['docs'], 
                        rag_type=setup_fixture['rag_type']['Standard'], 
                        chunk_method=setup_fixture['chunk_method'], 
                        chunk_size=setup_fixture['chunk_size'], 
                        chunk_overlap=setup_fixture['chunk_overlap'])
    assert result['rag'] == setup_fixture['rag_type']['Standard']
    assert result['pages'] is not None
    assert result['chunks'] is not None
    metadata_test={'source': 'test1.pdf', 'page': 1, 'start_index': 0}
    test_hash='e006e6fbafe375d1faff4783878c302a70c90ad9'
    assert _stable_hash_meta(result['chunks'][0].metadata) == _stable_hash_meta(metadata_test)  # Tests that the metadata is correct
    assert _stable_hash_meta(result['chunks'][0].metadata) == test_hash # Tests that the has is correct
    assert result['splitters'] is not None

# Test initialize database with a test query
def test_initialize_database_pinecone(monkeypatch,setup_fixture):
    '''
    Test the initialization of a Pinecone database.

    Args:
        monkeypatch: The monkeypatch fixture.
        setup_fixture (dict): The setup fixture containing the necessary parameters.

    Returns:
        PineconeVectorStore: The initialized Pinecone vector store.

    Raises:
        AssertionError: If the initialized vector store is not an instance of PineconeVectorStore.
    '''
    index_name = 'test-index'
    rag_type = 'Standard'
    clear = True
    init_ragatouille = False
    show_progress = False

    test_query_params={'index_type':'Pinecone',
                       'query_model': 'OpenAI', 
                       'embedding_name': 'text-embedding-ada-002'}
    query_model=parse_test_model('embedding', test_query_params, setup_fixture)

    # Test with environment variable local_db_path
    try:
        vectorstore = initialize_database(test_query_params['index_type'], 
                                          index_name, 
                                          query_model, 
                                          rag_type, 
                                          os.environ['LOCAL_DB_PATH'], 
                                          clear, 
                                          init_ragatouille, 
                                          show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
        raise e

    assert isinstance(vectorstore, PineconeVectorStore)
    delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    # Test with local_db_path set manually, show it doesn't work if not set
    monkeypatch.delenv('LOCAL_DB_PATH', raising=False)
    with pytest.raises(Exception):
        initialize_database(test_query_params['index_type'], 
                            index_name, 
                            query_model, 
                            rag_type, 
                            os.environ['LOCAL_DB_PATH'], 
                            clear, 
                            init_ragatouille, 
                            show_progress)
    try:    # Probably redudnant but to avoid cleanup
        delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
        raise e
    except:
        pass
def test_initialize_database_chromadb(monkeypatch,setup_fixture):
    '''
    Test the initialization of a Chroma database.

    Args:
        monkeypatch: The monkeypatch fixture.
        setup_fixture (dict): A dictionary containing setup fixtures.

    Returns:
        None

    Raises:
        AssertionError: If the vectorstore is not an instance of Chroma.

    '''
    index_name = 'test-index'
    rag_type = 'Standard'
    clear = True
    init_ragatouille = False
    show_progress = False

    test_query_params={'index_type':'ChromaDB',
                       'query_model': 'OpenAI', 
                       'embedding_name': 'text-embedding-ada-002'}
    query_model=parse_test_model('embedding', test_query_params, setup_fixture)

    # Test with environment variable local_db_path
    try:
        vectorstore = initialize_database(test_query_params['index_type'], 
                                          index_name, 
                                          query_model, 
                                          rag_type, 
                                          os.environ['LOCAL_DB_PATH'], 
                                          clear, 
                                          init_ragatouille, 
                                          show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
        raise e

    assert isinstance(vectorstore, Chroma)
    delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    # Test with local_db_path set manually, show it doesn't work if not set
    monkeypatch.delenv('LOCAL_DB_PATH', raising=False)
    with pytest.raises(Exception):
        initialize_database(test_query_params['index_type'], 
                            index_name, 
                            query_model, 
                            rag_type, 
                            os.environ['LOCAL_DB_PATH'], 
                            clear, 
                            init_ragatouille, 
                            show_progress)
    try:    # Probably redudnant but to avoid cleanup
        delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
        raise e
    except:
        pass
def test_initialize_database_ragatouille(monkeypatch,setup_fixture):
    '''
    Test the initialization of a database for RAGatouille.

    Args:
        monkeypatch: The monkeypatch fixture.
        setup_fixture (dict): A dictionary containing the setup fixture.

    Returns:
        None

    Raises:
        AssertionError: If the vectorstore is not an instance of RAGPretrainedModel.

    '''
    index_name = 'test-index'
    rag_type = 'Standard'
    clear = True
    init_ragatouille = True
    show_progress = False

    test_query_params={'index_type':'RAGatouille',
                    'query_model': 'RAGatouille', 
                    'embedding_name': 'colbert-ir/colbertv2.0'}
    query_model=parse_test_model('embedding', test_query_params, setup_fixture)

    # Test with environment variable local_db_path
    try:
        vectorstore = initialize_database(test_query_params['index_type'], 
                                          index_name, 
                                          query_model, 
                                          rag_type, 
                                          os.environ['LOCAL_DB_PATH'], 
                                          clear, 
                                          init_ragatouille, 
                                          show_progress)
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
        raise e
    
    assert isinstance(vectorstore, RAGPretrainedModel)
    delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])

    # Test with local_db_path set manually, show it doesn't work if not set
    monkeypatch.delenv('LOCAL_DB_PATH', raising=False)
    with pytest.raises(Exception):
        initialize_database(test_query_params['index_type'], 
                            index_name, 
                            query_model, 
                            rag_type, 
                            os.environ['LOCAL_DB_PATH'], 
                            clear, 
                            init_ragatouille, 
                            show_progress)
    try:    # Probably redudnant but to avoid cleanup
        delete_index(test_query_params['index_type'], index_name, rag_type, local_db_path=os.environ['LOCAL_DB_PATH'])
        raise e
    except:
        pass

# Test end to end process, adding query
def test_database_setup_and_query(test_input,setup_fixture):
    '''Tests the entire process of initializing a database, upserting documents, and deleting a database.

    Args:
        setup_fixture (dict): The setup fixture containing the necessary configuration for the test.
        test_input (str): The test input.

    Raises:
        Exception: If there is an error during the test.

    Returns:
        None
    '''
    test, print_str = parse_test_case(setup_fixture,test_input)
    index_name='test'+str(test['id'])
    print(f'Starting test: {print_str}')

    query_model=parse_test_model('embedding', test, setup_fixture)
    llm=parse_test_model('llm', test, setup_fixture)    

    try: 
        vectorstore = load_docs(
            test['index_type'],
            setup_fixture['docs'],
            rag_type=test['rag_type'],
            query_model=query_model,
            index_name=index_name, 
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            clear=True,
            batch_size=setup_fixture['batch_size'],
            local_db_path=setup_fixture['LOCAL_DB_PATH'],
            llm=llm)
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
            index_name = index_name + llm.model_name.replace('/', '-') + '-summary' 

        if test['index_type'] == 'RAGatouille':
            # query_model_qa=vectorstore  
            query_model_qa = RAGPretrainedModel.from_index(index_path=os.path.join(setup_fixture['LOCAL_DB_PATH'],
                                                                                   '.ragatouille/colbert/indexes',
                                                                                   index_name))         
        else:
            query_model_qa = query_model
        assert query_model_qa is not None
        
        qa_model_obj = QA_Model(test['index_type'],
                            index_name,
                            query_model_qa,
                            llm,
                            rag_type=test['rag_type'],
                            local_db_path=setup_fixture['LOCAL_DB_PATH'])
        print('QA model object created.')
        assert qa_model_obj is not None

        qa_model_obj.query_docs(setup_fixture['test_prompt'])
        assert qa_model_obj.ai_response is not None
        assert qa_model_obj.sources is not None
        assert qa_model_obj.memory is not None
        

        alternate_question = qa_model_obj.generate_alternative_questions(setup_fixture['test_prompt'])
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

    # Use the existing config file, to check they are set up correctly.
    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file=os.path.join(home_dir, 'config', 'config_admin.json')   # config_tester.json is a subset of config_admin, no need to test.

    # Test case: Only embeddings is True
    sidebar_config = load_sidebar(config_file=config_file, vector_database=True, embeddings=True)
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'OpenAI'

    # Test case: Only rag_type is True
    sidebar_config = load_sidebar(config_file=config_file, vector_database=True, rag_type=True)
    assert 'rag_type' in sidebar_config
    assert sidebar_config['rag_type'] == 'Standard'  

    # Test case: Only embeddings and rag_type are True
    sidebar_config = load_sidebar(config_file=config_file, vector_database=True, embeddings=True, rag_type=True)
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'OpenAI'

    # Test case: Only llm is True
    sidebar_config = load_sidebar(config_file=config_file, vector_database=True, llm=True)
    assert 'llm_source' in sidebar_config
    assert sidebar_config['llm_source'] == 'OpenAI'

    # Test case: Only model_options is True
    sidebar_config = load_sidebar(config_file=config_file, vector_database=True, model_options=True)
    assert 'temperature' in sidebar_config['model_options']
    assert sidebar_config['model_options']['temperature'] == 0.1
    assert 'output_level' in sidebar_config['model_options']
    assert sidebar_config['model_options']['output_level'] == 1000

    # Test case: All options are True
    sidebar_config = load_sidebar(config_file=config_file, vector_database=True,
                                  embeddings=True, rag_type=True, llm=True, model_options=True)
    assert 'index_type' in sidebar_config
    assert sidebar_config['index_type'] == 'ChromaDB'
    assert 'query_model' in sidebar_config
    assert sidebar_config['query_model'] == 'OpenAI'
    assert 'rag_type' in sidebar_config
    assert sidebar_config['rag_type'] == 'Standard' 
    assert 'llm_source' in sidebar_config
    assert sidebar_config['llm_source'] == 'OpenAI'
    assert 'temperature' in sidebar_config['model_options']
    assert sidebar_config['model_options']['temperature'] == 0.1
    assert 'output_level' in sidebar_config['model_options']
    assert sidebar_config['model_options']['output_level'] == 1000
def test_set_secrets_with_environment_variables(monkeypatch):
    '''
    Test case to verify the behavior of the set_secrets function when environment variables are set.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables during testing.

    Returns:
        None

    Raises:
        AssertionError: If the secrets are not set correctly.
    '''
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
    '''
    Test case for the set_secrets function with sidebar data.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables.

    Returns:
        None

    Raises:
        AssertionError: If the secrets are not set correctly.
    '''
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
@pytest.mark.parametrize('missing_key',
                         ['OPENAI_API_KEY','VOYAGE_API_KEY','PINECONE_API_KEY','HUGGINGFACEHUB_API_TOKEN'])
def test_set_secrets_missing_api_keys(monkeypatch, missing_key):
    '''
    Test case for setting secrets with missing API keys.

    Args:
        monkeypatch: A pytest fixture for patching values during testing.
        missing_key: The missing API key to be tested.

    Raises:
        SecretKeyException: If the set_secrets function raises an exception.

    Returns:
        None
    '''
    print(f'Testing missing required key: {missing_key}')
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
    '''
    Test case for the `st_setup_page` function when only the local db path is defined.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables during testing.

    Returns:
        None
    '''
    page_title = 'Test Page'

    # Clear all environment variables
    for var in list(os.environ.keys()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr('admin.load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setattr('admin.find_dotenv', lambda *args, **kwargs: '')  # Assuming an empty string simulates not finding a .env file
    monkeypatch.setenv('LOCAL_DB_PATH', os.path.abspath(os.path.dirname(__file__))) # Set the local db path to this directory

    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file=os.path.join(home_dir, 'config', 'config_admin.json')   # config_tester.json is a subset of config_admin, no need to test.

    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir, config_file)

    # Assert
    assert paths['db_folder_path'] == os.getenv('LOCAL_DB_PATH')
    assert sb == {}
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': None,
                       'OPENAI_API_KEY': None,
                       'PINECONE_API_KEY': None,
                       'VOYAGE_API_KEY': None}
def test_st_setup_page_local_db_path_not_defined(monkeypatch):
    '''
    Test case to verify the behavior of st_setup_page function when LOCAL_DB_PATH is not defined.

    Args:
        monkeypatch: Monkeypatch object for modifying environment variables.

    Returns:
        None
    '''

    page_title = 'Test Page'

    # Clear all environment variables
    for var in list(os.environ.keys()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr('admin.load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setattr('admin.find_dotenv', lambda *args, **kwargs: '')  # Assuming an empty string simulates not finding a .env file
    monkeypatch.setenv('LOCAL_DB_PATH', None) # Set to none

    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file=os.path.join(home_dir, 'config', 'config_admin.json')   # config_tester.json is a subset of config_admin, no need to test.

    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir, config_file)

    # Assert
    assert paths['db_folder_path'] == 'None'
    assert sb == {}
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': None,
                       'OPENAI_API_KEY': None,
                       'PINECONE_API_KEY': None,
                       'VOYAGE_API_KEY': None}
def test_st_setup_page_local_db_path_w_all_man_input(monkeypatch):
    '''
    Test case for the st_setup_page function with all inputs in sidebar and manual input for environment variables.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables and other attributes during testing.

    Returns:
        None
    '''

    page_title = 'Test Page'
    sidebar_config = {
        'vector_database': True,
        'embeddings': True,
        'rag_type': True,
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
    config_file=os.path.join(home_dir, 'config', 'config_admin.json')   # config_tester.json is a subset of config_admin, no need to test.

    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir, config_file, sidebar_config)

    # Assert
    assert paths['db_folder_path'] == os.getenv('LOCAL_DB_PATH')
    assert isinstance(sb, dict) and sb != {}    # Test that it's not an empty dictionary
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                       'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                       'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
                       'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY')}
def test_st_setup_page_local_db_path_w_all_env_input(monkeypatch,temp_dotenv):
    """
    Test case for the `st_setup_page` function with all environment inputs.

    Args:
        monkeypatch: A monkeypatch object for modifying environment variables.
        temp_dotenv: The path to the temporary dotenv file.

    Returns:
        None
    """

    page_title = 'Test Page'
    sidebar_config = {
        'vector_database': True,
        'embeddings': True,
        'rag_type': True,
        'llm': True,
        'model_options': True
    }

    # Clear all environment variables, simulate .env load in st_setup_page and pre-set local_db_path
    for var in list(os.environ.keys()):
        monkeypatch.delenv(var, raising=False)
    dotenv_path = temp_dotenv

    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file=os.path.join(home_dir, 'config', 'config_admin.json')   # config_tester.json is a subset of config_admin, no need to test.


    # Act
    paths, sb, secrets = st_setup_page(page_title, home_dir, config_file, sidebar_config)

    # Assert
    assert paths['db_folder_path'] == os.getenv('LOCAL_DB_PATH')
    assert isinstance(sb, dict) and sb != {}    # Test that it's not an empty dictionary
    assert secrets == {'HUGGINGFACEHUB_API_TOKEN': os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                       'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                       'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
                       'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY')}

# Test data visualization
def test_get_docs_df(setup_fixture):
    """
    Test case for the get_docs_df function.

    Args:
        setup_fixture (dict): The setup fixture containing necessary parameters.

    Returns:
        None
    """
    index_name = 'test-index'
    test_query_params={'index_type':'ChromaDB',
                       'query_model': 'OpenAI', 
                       'embedding_name': 'text-embedding-ada-002'}
    query_model=parse_test_model('embedding', test_query_params, setup_fixture)

    # Call the function
    # TODO add pinecone test
    df = get_docs_df('ChromaDB',setup_fixture['LOCAL_DB_PATH'], index_name, query_model)

    # Perform assertions
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns
    assert "source" in df.columns
    assert "page" in df.columns
    assert "document" in df.columns
    assert "embedding" in df.columns
def test_get_questions_df(setup_fixture):
    """
    Test case for the get_questions_df function.

    Args:
        setup_fixture (dict): The setup fixture containing necessary data for the test.

    Returns:
        None
    """
    index_name = 'test-index'
    test_query_params={'index_type':'ChromaDB',
                       'query_model': 'OpenAI', 
                       'embedding_name': 'text-embedding-ada-002'}
    query_model=parse_test_model('embedding', test_query_params, setup_fixture)

    # Call the function
    df = get_questions_df(setup_fixture['LOCAL_DB_PATH'], index_name, query_model)

    # Perform assertions
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns
    assert "question" in df.columns
    assert "answer" in df.columns
    assert "sources" in df.columns
    assert "embedding" in df.columns
def test_get_docs_questions_df(setup_fixture):
    """
    Test function for the get_docs_questions_df() method.

    Args:
        setup_fixture: The setup fixture containing necessary parameters for the test.

    Raises:
        Exception: If there is an error, the function will raise an exception.

    Returns:
        None
    """
    
    index_name='test-vizualisation'
    rag_type='Standard'
    test_query_params={'index_type':'ChromaDB',
                       'query_model': 'OpenAI', 
                       'embedding_name': "text-embedding-3-large"}
    test_llm_params={'llm_family': 'OpenAI', 
                     'llm': 'gpt-3.5-turbo-0125'}
    query_model=parse_test_model('embedding', test_query_params, setup_fixture)
    llm=parse_test_model('llm', test_llm_params, setup_fixture)

    try: 
        vectorstore = load_docs(
            test_query_params['index_type'],
            setup_fixture['docs'],
            rag_type=rag_type,
            query_model=query_model,
            index_name=index_name, 
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            clear=True,
            batch_size=setup_fixture['batch_size'],
            local_db_path=setup_fixture['LOCAL_DB_PATH'],
            llm=llm)
        qa_model_obj = QA_Model(test_query_params['index_type'],
                            index_name,
                            query_model,
                            llm,
                            rag_type=rag_type,
                            local_db_path=setup_fixture['LOCAL_DB_PATH'])
        qa_model_obj.query_docs(setup_fixture['test_prompt'])
        assert qa_model_obj.query_vectorstore is not None
        
        df = get_docs_questions_df(
            setup_fixture['LOCAL_DB_PATH'],
            index_name,
            setup_fixture['LOCAL_DB_PATH'],
            index_name+'-queries',
            query_model
        )

        # Assert the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "id" in df.columns
        assert "source" in df.columns
        assert "page" in df.columns
        assert "document" in df.columns
        assert "embedding" in df.columns
        assert "type" in df.columns
        assert "num_sources" in df.columns
        assert "first_source" in df.columns
        assert "used_by_questions" in df.columns
        assert "used_by_num_questions" in df.columns
        assert "used_by_question_first" in df.columns

        delete_index(test_query_params['index_type'],
                index_name, 
                rag_type,
                local_db_path=setup_fixture['LOCAL_DB_PATH'])

    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test_query_params['index_type'],
                index_name, 
                rag_type,
                local_db_path=setup_fixture['LOCAL_DB_PATH'])
        raise e
def test_add_clusters(setup_fixture):
    """
    Test function for the add_clusters function.

    Args:
        setup_fixture (dict): A dictionary containing setup fixtures for the test.

    Returns:
        None
    """
    
    index_name='test-vizualisation'
    rag_type='Standard'
    test_query_params={'index_type':'ChromaDB',
                       'query_model': 'OpenAI', 
                       'embedding_name': "text-embedding-3-large"}
    test_llm_params={'llm_family': 'OpenAI', 
                     'llm': 'gpt-3.5-turbo-0125'}
    query_model=parse_test_model('embedding', test_query_params, setup_fixture)
    llm=parse_test_model('llm', test_llm_params, setup_fixture)

    try: 
        vectorstore = load_docs(
            test_query_params['index_type'],
            setup_fixture['docs'],
            rag_type=rag_type,
            query_model=query_model,
            index_name=index_name, 
            chunk_size=setup_fixture['chunk_size'],
            chunk_overlap=setup_fixture['chunk_overlap'],
            clear=True,
            batch_size=setup_fixture['batch_size'],
            local_db_path=setup_fixture['LOCAL_DB_PATH'],
            llm=llm)
        qa_model_obj = QA_Model(test_query_params['index_type'],
                            index_name,
                            query_model,
                            llm,
                            rag_type=rag_type,
                            local_db_path=setup_fixture['LOCAL_DB_PATH'])
        qa_model_obj.query_docs(setup_fixture['test_prompt'])

        df = get_docs_questions_df(
            setup_fixture['LOCAL_DB_PATH'],
            index_name,
            setup_fixture['LOCAL_DB_PATH'],
            index_name+'-queries',
            query_model
        )

        delete_index(test_query_params['index_type'],
                index_name, 
                rag_type,
                local_db_path=setup_fixture['LOCAL_DB_PATH'])

    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test_query_params['index_type'],
                index_name, 
                rag_type,
                local_db_path=setup_fixture['LOCAL_DB_PATH'])

    # Check the add_clusters function with no labeling
    n_clusters = 2  # Define the expected number of clusters
    df_with_clusters = add_clusters(df, n_clusters) # Call the add_clusters function
    assert len(df_with_clusters["Cluster"].unique()) == n_clusters  # Check if the number of clusters is correct
    for cluster in df_with_clusters["Cluster"].unique():    # Check if the number of documents per cluster is correct
        num_documents = len(df_with_clusters[df_with_clusters["Cluster"] == cluster])
        assert num_documents >= 1

    # Check the add_clusters function with labeling
    n_clusters = 2  # Define a different number of clusters
    df_with_clusters = add_clusters(df, n_clusters, llm, 2)  # Call the add_clusters function
    assert len(df_with_clusters["Cluster"].unique()) == n_clusters  # Check if the number of clusters is correct
    assert "Cluster_Label" in df_with_clusters.columns  # Check if the cluster labels are added correctly
    assert df_with_clusters["Cluster_Label"].notnull().all()  # Check if the cluster labels are non-empty
    assert df_with_clusters["Cluster_Label"].apply(lambda x: isinstance(x, str)).all()  # Check if the cluster labels are strings
    for cluster in df_with_clusters["Cluster"].unique():  # Check if the number of documents per cluster is more than zero
        num_documents = len(df_with_clusters[df_with_clusters["Cluster"] == cluster])
        assert num_documents > 0