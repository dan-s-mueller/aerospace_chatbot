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
    """
    Permute test data to generate all possible combinations.

    Args:
        test_data (list[dict]): A list of dictionaries where each dictionary represents a set of keys and values to be iterated.

    Returns:
        list[dict]: A list of dictionaries representing all possible combinations of the input test data.

    Example:
        test_data = [{'index_type': ['ChromaDB'], 'rag_type': ['Standard', 'Parent-Child', 'Summary']}, 
                     {'index_type': ['Pinecone'], 'rag_type': ['Standard', 'Parent-Child']}]
        permute_tests(test_data) returns:
                    [{'index_type': 'ChromaDB', 'rag_type': 'Standard'},
                    {'index_type': 'ChromaDB', 'rag_type': 'Parent-Child'},
                    {'index_type': 'ChromaDB', 'rag_type': 'Summary'},
                    {'index_type': 'Pinecone', 'rag_type': 'Standard'},
                    {'index_type': 'Pinecone', 'rag_type': 'Parent-Child'}]
    """
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

def custom_encoder(obj):
    """
    Converts non-serializable objects to a printable string.

    Args:
        obj: The object to be encoded.

    Returns:
        str: The encoded string representation of the object.
    """
    if isinstance(obj, ChatOpenAI):
        return obj.model_name
    elif not isinstance(obj, str):
        return str(type(obj))
    return str(obj)

def generate_test_cases(setup, export:bool=True):
    """
    Generates test cases for screening.

    Args:
        export (bool, optional): Indicates whether to export the test cases to a JSON file. Defaults to False.

    Returns:
        list: A list of test cases.

    Raises:
        None

    Example:
        test_cases = generate_test_cases(export=True)
    """

    # Determine the set of cases to screen
    # TODO throw in bad inputs for each of the 4 major types below.
    test_cases = [
        {
            # Tests ChromaDB setups, advanced RAG (standard/parent-child)
            'index_type': [setup['index_type']['ChromaDB']],
            'query_model': [setup['query_model']['OpenAI']],
            'rag_type': [setup['rag_type']['Standard'], setup['rag_type']['Parent-Child']],
            'llm': [setup['llm']['Hugging Face']]
        },
        {
            # Tests advanced RAG (summary) and LLM (openai/hugging face)
            'index_type': [setup['index_type']['ChromaDB']],
            'query_model': [setup['query_model']['OpenAI']],
            'rag_type': [setup['rag_type']['Summary']],
            'llm': [setup['llm']['OpenAI'], setup['llm']['Hugging Face']]
        },
        {
            # Tests Pinecone setups, embedding types (openai/voyage)
            'index_type': [setup['index_type']['Pinecone']],
            'query_model': [setup['query_model']['OpenAI'], setup['query_model']['Voyage']],
            'rag_type': [setup['rag_type']['Standard']],
            'llm': [setup['llm']['Hugging Face']]
        },
        {
            # Tests RAGatouille setup
            'index_type': [setup['index_type']['RAGatouille']],
            'query_model': [setup['query_model']['RAGatouille']],
            'rag_type': [setup['rag_type']['Standard']],
            'llm': [setup['llm']['Hugging Face']]
        }
    ]

    tests = permute_tests(test_cases)
    
    if export:
        file_path = 'test_cases.json'
        with open(file_path, 'w') as json_file:
            json.dump(tests, json_file, default=custom_encoder, indent=4)
    
    return tests

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

    tests=generate_test_cases(setup)

    return setup, tests

# Define a fixture for each test case
# @pytest.fixture(params=[0])  # This will be replaced dynamically
# def test_case(request, setup_fixture):
#     _, tests = setup_fixture
#     return tests[request.param]


def test_env_variables_exist(setup_fixture):
    """
    Test case to check if the required environment variables exist.

    This test case checks if the following variables exist:
    - OPENAI_API_KEY
    - VOYAGE_API_KEY
    - HUGGINGFACEHUB_API_TOKEN
    - PINECONE_API_KEY

    If any of these variables are None, the test will fail.

    Returns:
        None
    """
    setup, _ = setup_fixture

    assert setup['OPENAI_API_KEY'] is not None
    assert setup['VOYAGE_API_KEY'] is not None
    assert setup['HUGGINGFACEHUB_API_TOKEN'] is not None
    assert setup['PINECONE_API_KEY'] is not None
    print('Environment variables test passed.')

def test_chunk_docs(setup_fixture):
    """
    Test the chunk_docs function.

    This method tests the chunk_docs function with different rag types and verifies the expected outputs.

    Returns:
        None
    """
    setup, _ = setup_fixture

    # Test case 1: Standard rag
    result = chunk_docs(setup['docs'], 
                        rag_type=setup['rag_type']['Standard'], 
                        chunk_method=setup['chunk_method'], 
                        chunk_size=setup['chunk_size'], 
                        chunk_overlap=setup['chunk_overlap'])
    assert result['rag'] == setup['rag_type']['Standard']
    assert result['pages'] is not None
    assert result['chunks'] is not None
    assert result['splitters'] is not None
    print('Standard rag test passed.')

    # Test case 2: Parent-Child rag
    result = chunk_docs(setup['docs'], 
                        rag_type=setup['rag_type']['Parent-Child'], 
                        chunk_method=setup['chunk_method'], 
                        chunk_size=setup['chunk_size'], 
                        chunk_overlap=setup['chunk_overlap'])
    assert result['rag'] == setup['rag_type']['Parent-Child']
    assert result['pages']['doc_ids'] is not None
    assert result['pages']['parent_chunks'] is not None
    assert result['chunks'] is not None
    assert result['splitters']['parent_splitter'] is not None
    assert result['splitters']['child_splitter'] is not None
    print('Parent-child rag test passed.')

    # Test case 3: Summary rag
    result = chunk_docs(setup['docs'], 
                        rag_type=setup['rag_type']['Summary'], 
                        chunk_method=setup['chunk_method'], 
                        chunk_size=setup['chunk_size'], 
                        chunk_overlap=setup['chunk_overlap'], 
                        llm=setup['llm']['Hugging Face'])
    assert result['rag'] == setup['rag_type']['Summary']
    assert result['pages']['doc_ids'] is not None
    assert result['pages']['docs'] is not None
    assert result['summaries'] is not None
    assert result['llm'] == setup['llm']['Hugging Face']
    print('Summary rag test passed.')

# TODO add parematerized tests for the process function
def test_database_setup_and_query(setup_fixture):
    """  
    Tests the entire process of initializing a database, upserting documents, and deleting a database
    """
    setup, tests = setup_fixture
    # test=tests[0]
    test=tests[0]

    print(f"Starting test {test['id']}:")

    try:            
        chunker = chunk_docs(setup['docs'], 
                            rag_type=test['rag_type'], 
                            chunk_method=setup['chunk_method'], 
                            chunk_size=setup['chunk_size'], 
                            chunk_overlap=setup['chunk_overlap'],
                            llm=test['llm'])
        assert chunker is not None
        print('Docs chunked.')
        
        vectorstore = initialize_database(test['index_type'], 
                                        'test'+str(test['id']), 
                                        test['query_model'],
                                        test['rag_type'],
                                        local_db_path=setup['LOCAL_DB_PATH'], 
                                        clear=False,
                                        test_query=False,
                                        init_ragatouille=True)
        assert vectorstore is not None
        print('Database initialized.')
        
        vectorstore, retriever = upsert_docs(test['index_type'], 
                                            'test'+str(test['id']), 
                                            vectorstore,
                                            chunker,
                                            batch_size=setup['batch_size'], 
                                            local_db_path=setup['LOCAL_DB_PATH'])
        assert vectorstore is not None
        assert retriever is not None
        print('Docs upserted.')
        
        if test['index_type'] == 'RAGatouille':
            query_model_qa = RAGPretrainedModel.from_index(setup['LOCAL_DB_PATH']+'/.ragatouille/colbert/indexes/'+'test'+str(test['id']),
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
                            local_db_path=os.path.join(setup['LOCAL_DB_PATH'],'ref_dbs'))
        logging.info('QA model object created.')

        assert qa_model_obj is not None
        qa_model_obj.query_docs(setup['test_prompt'])
        response = qa_model_obj.ai_response
        assert response is not None
        alternate_question = qa_model_obj.generate_alternative_questions(setup['test_prompt'], response=response)
        assert alternate_question is not None

        # print(f'Query response: {response}')
        # print(f'Alternate questions: {alternate_question}')
        print('Query and alternative question successful!')

        delete_index(test['index_type'],
                        'test'+str(test['id']), 
                        test['rag_type'],
                        local_db_path=setup['LOCAL_DB_PATH'])
        print('Database deleted.')
    except Exception as e:  # If there is an error, be sure to delete the database
        delete_index(test['index_type'],
                        'test'+str(test['id']), 
                        test['rag_type'],
                        local_db_path=setup['LOCAL_DB_PATH'])
        raise e
        
# TODO add tests for path setup. test config, data, db paths
# TODO add tests for subfunctions admin and data_processing