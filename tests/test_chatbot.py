import os, sys, json
import itertools
import pytest
import pandas as pd
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

import nltk # Do before ragatioulle import to avoid logs
nltk.download('punkt', quiet=True)
from ragatouille import RAGPretrainedModel

import chromadb

# Import local variables
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src/aerospace_chatbot'))
from data_processing import chunk_docs, initialize_database, load_docs, \
      delete_index, _stable_hash_meta, get_docs_df, get_questions_df, \
      add_clusters, get_docs_questions_df
from admin import SidebarManager, set_secrets, SecretKeyException, DatabaseException
from queries import QA_Model

# TODO add tests to check conversation history functionality

# Functions
def permute_tests(test_data):
    '''Generate permutations of test cases.'''
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
def read_test_cases(json_path:str):
    with open(json_path, 'r') as json_file:
        test_cases = json.load(json_file)
    return test_cases
def pytest_generate_tests(metafunc):
    '''
    Use pytest_generate_tests to dynamically generate tests.
    Tests generates tests from a static file (test_cases.json). See test_cases.json for more details.
    '''
    if 'test_input' in metafunc.fixturenames:
        tests = read_test_cases(os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_cases.json'))
        metafunc.parametrize('test_input', tests)
def parse_test_case(setup,test_case):
    ''' Parse test case to be used in the test functions.'''
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
    """Parses the test model based on the given type and test parameters."""
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
        elif test['llm_family'] == 'Anthropic':
            llm = ChatAnthropic(model=test['llm'], anthropic_api_key=setup_fixture['ANTHROPIC_API_KEY'], max_tokens=500)
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
    """Sets up the fixture for testing the backend."""
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
    """Creates a temporary .env file for testing purposes."""
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
    '''Test the chunk_docs function with standard RAG.'''
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
    """Test case for the `chunk_docs` function with no chunking and merging."""

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
    '''Test the chunk_docs function with no chunking.'''
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
    '''Test the chunk_docs function with parent-child RAG.'''
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
    '''Test the chunk_docs function with summary RAG.'''
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
    '''Test case for chunk_id_lookup function.'''
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
    '''Test the initialization of a Pinecone database.'''
    index_name = 'test-index'
    rag_type = 'Standard'
    clear = True
    init_ragatouille = False
    show_progress = False

    test_query_params={'index_type':'Pinecone',
                       'query_model': 'OpenAI', 
                       'embedding_name': 'text-embedding-3-large'}
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
    ''' Test the initialization of a Chroma database.'''
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
    '''Test the initialization of a database for RAGatouille.'''
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
    '''Tests the entire process of initializing a database, upserting documents, and deleting a database.'''
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
def test_sidebar_manager():
    """Test the SidebarManager class functionality."""
    # Setup
    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file = os.path.join(home_dir, 'config', 'config_admin.json')

    # Test initialization
    sidebar_manager = SidebarManager(config_file)
    assert sidebar_manager.config is not None
    assert 'databases' in sidebar_manager.config
    assert 'embeddings' in sidebar_manager.config
    assert 'llms' in sidebar_manager.config
    assert 'rag_types' in sidebar_manager.config

    # Define test cases
    test_cases = [
        {'vector_database': True, 'embeddings': True},  # Only embeddings is True
        {'vector_database': True, 'rag_type': True},    # Only rag_type is True
        {'vector_database': True, 'embeddings': True, 'rag_type': True},  # Only embeddings and rag_type are True
        {'vector_database': True, 'llm': True},         # Only llm is True
        {'vector_database': True, 'model_options': True},  # Only model_options is True
        {'vector_database': True, 'embeddings': True, 'rag_type': True, 'llm': True, 'model_options': True}  # All options are True
    ]

    for case in test_cases:
        # Render sidebar with specific configuration
        sb_out = sidebar_manager.render_sidebar(**case)
        
        # Verify core dependencies are always set
        assert 'index_type' in sb_out
        assert 'embedding_name' in sb_out
        assert 'rag_type' in sb_out

        # Verify specific outputs based on enabled features
        if case.get('embeddings'):
            assert 'query_model' in sb_out
            assert sb_out['query_model'] == 'OpenAI'
            assert 'embedding_name' in sb_out

        if case.get('rag_type'):
            assert 'rag_type' in sb_out
            assert sb_out['rag_type'] == 'Standard'

        if case.get('llm'):
            assert 'llm_source' in sb_out
            assert sb_out['llm_source'] == 'OpenAI'
            assert 'llm_model' in sb_out

        if case.get('model_options'):
            assert 'model_options' in sb_out
            assert 'temperature' in sb_out['model_options']
            assert sb_out['model_options']['temperature'] == 0.1
            assert 'output_level' in sb_out['model_options']
            assert sb_out['model_options']['output_level'] == 1000
            assert 'k' in sb_out['model_options']
            if sb_out['index_type'] != 'RAGatouille':
                assert 'search_type' in sb_out['model_options']

        # Additional verification for the "all options" case
        if all(key in case for key in ['embeddings', 'rag_type', 'llm', 'model_options']):
            assert sb_out['index_type'] == 'ChromaDB'
            assert sb_out['query_model'] == 'OpenAI'
            assert sb_out['rag_type'] == 'Standard'
            assert sb_out['llm_source'] == 'OpenAI'

    # Test paths functionality
    paths = sidebar_manager.get_paths(home_dir)
    assert 'base_folder_path' in paths
    assert 'db_folder_path' in paths
    assert 'data_folder_path' in paths
def test_set_secrets_with_environment_variables(monkeypatch):
    '''Test case to verify the behavior of the set_secrets function when environment variables are set.'''
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
    '''Test case for the set_secrets function with sidebar data.'''
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
    '''Test case for setting secrets with missing API keys.'''
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
    assert os.path.exists(paths['data_folder_path'])
    
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
    with pytest.raises(KeyError):
        manager._validate_config(invalid_config)
    
    # Test with empty config
    with pytest.raises(KeyError):
        manager._validate_config({})
def test_sidebar_manager_render_sidebar_edge_cases():
    """Test render_sidebar method with edge cases"""
    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file = os.path.join(home_dir, 'config', 'config_admin.json')
    
    manager = SidebarManager(config_file)
    
    # Test with no options enabled
    sb_out = manager.render_sidebar(vector_database=False, 
                                  embeddings=False, 
                                  rag_type=False, 
                                  llm=False, 
                                  model_options=False)
    assert 'index_type' in sb_out
    assert 'embedding_name' in sb_out
    assert 'rag_type' in sb_out
    
    # Test with invalid combination (embeddings=True but vector_database=False)
    sb_out = manager.render_sidebar(vector_database=False, 
                                  embeddings=True)
    assert 'query_model' not in sb_out
    
    # Test with all options enabled but empty config sections
    # Create temporary config with empty sections
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tf:
        empty_config = {
            'databases': {},
            'embeddings': {},
            'llms': {},
            'rag_types': {}
        }
        json.dump(empty_config, tf)
        tf.flush()
        empty_manager = SidebarManager(tf.name)
        sb_out = empty_manager.render_sidebar(vector_database=True, 
                                            embeddings=True, 
                                            rag_type=True, 
                                            llm=True, 
                                            model_options=True)
        assert sb_out == {}
def test_sidebar_manager_config_dependencies():
    """Test that config dependencies are properly enforced"""
    home_dir = os.path.abspath(os.path.dirname(__file__))
    home_dir = os.path.join(home_dir, '..')
    home_dir = os.path.normpath(home_dir)
    config_file = os.path.join(home_dir, 'config', 'config_admin.json')
    
    manager = SidebarManager(config_file)
    
    # Test that llm options require llm to be enabled
    sb_out = manager.render_sidebar(vector_database=True, 
                                  llm=False, 
                                  model_options=True)
    assert 'temperature' not in sb_out.get('model_options', {})
    
    # Test that embedding options require vector_database to be enabled
    sb_out = manager.render_sidebar(vector_database=False, 
                                  embeddings=True)
    assert 'query_model' not in sb_out

# Test data visualization
def test_get_docs_df(setup_fixture):
    """Test case for the get_docs_df function."""
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
    """Test case for the get_questions_df function."""
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
    """Test function for the get_docs_questions_df() method."""
    
    index_name='test-vizualisation'
    rag_type='Standard'
    test_query_params={'index_type':'ChromaDB',
                       'query_model': 'OpenAI', 
                       'embedding_name': 'text-embedding-ada-002'}
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
    """Test function for the add_clusters function."""
    
    index_name='test-vizualisation'
    rag_type='Standard'
    test_query_params={'index_type':'ChromaDB',
                       'query_model': 'OpenAI', 
                       'embedding_name': 'text-embedding-ada-002'}
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