import os, sys, logging, json
import itertools
import pytest
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from langchain_openai import ChatOpenAI

from ragatouille import RAGPretrainedModel

# Import local variables
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src/aerospace_chatbot'))
from data_processing import chunk_docs, initialize_database, upsert_docs, delete_index
from queries import QA_Model

def permute_tests(test_data):
    rows = []
    idx=1
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
        tests = read_test_cases('test_cases.json')
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
    # Logger
    logging.basicConfig(filename='test.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    
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

    LOCAL_DB_PATH='.'   # Default to the test path for easy cleanup.
    
    # Fixed inputs
    docs = ['test1.pdf', 'test2.pdf']
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
                    'Voyage':VoyageEmbeddings(voyage_api_key=VOYAGE_API_KEY),
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

def test_env_variables_exist(setup_fixture):
    assert setup_fixture['OPENAI_API_KEY'] is not None
    assert setup_fixture['VOYAGE_API_KEY'] is not None
    assert setup_fixture['HUGGINGFACEHUB_API_TOKEN'] is not None
    assert setup_fixture['PINECONE_API_KEY'] is not None
    print('Environment variables test passed.')

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

def test_database_setup_and_query(setup_fixture,test_input):
    """  
    Tests the entire process of initializing a database, upserting documents, and deleting a database
    """
    test, print_str =parse_test_case(setup_fixture,test_input)

    print(f"Starting test: {print_str}")

    try:            
        chunker = chunk_docs(setup_fixture['docs'], 
                            rag_type=test['rag_type'], 
                            chunk_method=setup_fixture['chunk_method'], 
                            chunk_size=setup_fixture['chunk_size'], 
                            chunk_overlap=setup_fixture['chunk_overlap'],
                            llm=test['llm'])
        assert chunker is not None
        print('Docs chunked.')
        
        vectorstore = initialize_database(test['index_type'], 
                                        'test'+str(test['id']), 
                                        test['query_model'],
                                        test['rag_type'],
                                        local_db_path=setup_fixture['LOCAL_DB_PATH'], 
                                        clear=False,
                                        test_query=False,
                                        init_ragatouille=True)
        assert vectorstore is not None
        print('Database initialized.')
        
        vectorstore, retriever = upsert_docs(test['index_type'], 
                                            'test'+str(test['id']), 
                                            vectorstore,
                                            chunker,
                                            batch_size=setup_fixture['batch_size'], 
                                            local_db_path=setup_fixture['LOCAL_DB_PATH'])
        assert vectorstore is not None
        assert retriever is not None
        print('Docs upserted.')
        
        if test['index_type'] == 'RAGatouille':
            query_model_qa = RAGPretrainedModel.from_index(setup_fixture['LOCAL_DB_PATH']+'/.ragatouille/colbert/indexes/'+'test'+str(test['id']),
                                                        n_gpu=0,
                                                        verbose=1)             
            print('RAGatouille query model created.')
        else:
            query_model_qa = test['query_model']
            print('Query model created.')
        assert query_model_qa is not None
        
        qa_model_obj = QA_Model(test['index_type'],
                            'test'+str(test['id']),
                            query_model_qa,
                            test['llm'],
                            rag_type=test['rag_type'],
                            test_query=False,
                            local_db_path=os.path.join(setup_fixture['LOCAL_DB_PATH'],'ref_dbs'))
        logging.info('QA model object created.')

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
# TODO add tests for subfunctions admin and data_processing