import unittest
import os
import sys
import logging
import json
import itertools
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from langchain_openai import ChatOpenAI

from ragatouille import RAGPretrainedModel

# Import local variables
sys.path.append('../src')
from data_processing import chunk_docs, initialize_database, upsert_docs, delete_index
from queries import QA_Model

# Logger
logging.basicConfig(filename='test.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

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
    for row_data in test_data:
        keys = row_data.keys()
        values = row_data.values()
        permutations = list(itertools.product(*values))
        for perm in permutations:
            row = dict(zip(keys, perm))
            rows.append(row)
    return rows
def custom_encoder(obj):
    """
    Converts non-serializable objects to a printable string.

    Args:
        obj: The object to be encoded.

    Returns:
        str: The encoded string representation of the object.
    """
    if not isinstance(obj, str):
        return str(type(obj))
    return str(obj)

class TestChatbot(unittest.TestCase):
    """
    A class that contains unit tests for the Chatbot functionality.

    This class inherits from the `unittest.TestCase` class and provides test cases for various aspects of the Chatbot implementation.

    Attributes:
        OPENAI_API_KEY (str): The API key for OpenAI.
        VOYAGE_API_KEY (str): The API key for Voyage.
        HUGGINGFACEHUB_API_TOKEN (str): The API token for Hugging Face Hub.
        PINECONE_API_KEY (str): The API key for Pinecone.
        LOCAL_DB_PATH (str): The local path for the database.

    Methods:
        setUp(): Sets up the test environment before each test case is executed.
        test_variables_exist(): Test case to check if the required variables exist.
        test_chunk_docs(): Test the chunk_docs function.
        generate_test_cases(export: bool = False): Generates test cases for screening.
        test_process(): Tests the entire process of initializing a database, upserting documents, and querying the database.
    """
    def setUp(self):
        """Sets up the test environment before each test case is executed.

        This method is called before each test case is executed to set up the necessary environment
        and variables for the tests.

        Args:
            None

        Returns:
            None
        """
        # Pull api keys from .env file. If these do not exist, create a .env file in the root directory and add the following.
        # TODO: Add functionality to input these keys as arguments to the test.
        load_dotenv(find_dotenv(),override=True)
        self.OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
        self.VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
        self.HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self.PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
        self.LOCAL_DB_PATH='.'   # Default to the test path for easy cleanup.
        
        # Fixed inputs
        self.docs = ['test1.pdf', 'test2.pdf']
        self.chunk_method='character_recursive'
        self.chunk_size=500
        self.chunk_overlap=0
        self.test_prompt='What are some nuances associated with the analysis and design of hinged booms?'   # Info on test2.pdf

        # Variable inputs
        self.llm={'OpenAI':ChatOpenAI(model_name='gpt-3.5-turbo-1106', # Openai
                                      openai_api_key=self.OPENAI_API_KEY,
                                      max_tokens=500), 
                  'Hugging Face':ChatOpenAI(base_url='https://api-inference.huggingface.co/v1',  # Hugging face
                                            model='mistralai/Mistral-7B-Instruct-v0.2',
                                            api_key=self.HUGGINGFACEHUB_API_TOKEN,
                                            max_tokens=500)}
        self.query_model={'OpenAI':OpenAIEmbeddings(model='text-embedding-ada-002',openai_api_key=self.OPENAI_API_KEY),
                          'Voyage':VoyageEmbeddings(voyage_api_key=self.VOYAGE_API_KEY),
                          'RAGatouille':'colbert-ir/colbertv2.0'}
        self.index_type = {index: index for index in ['ChromaDB', 'Pinecone', 'RAGatouille']}
        self.rag_type = {rag: rag for rag in ['Standard','Parent-Child','Summary']}

    def test_variables_exist(self):
        """
        Test case to check if the required variables exist.

        This test case checks if the following variables exist:
        - OPENAI_API_KEY
        - VOYAGE_API_KEY
        - HUGGINGFACEHUB_API_TOKEN
        - PINECONE_API_KEY

        If any of these variables are None, the test will fail.

        Returns:
            None
        """
        self.assertIsNotNone(self.OPENAI_API_KEY)
        self.assertIsNotNone(self.VOYAGE_API_KEY)
        self.assertIsNotNone(self.HUGGINGFACEHUB_API_TOKEN)
        self.assertIsNotNone(self.PINECONE_API_KEY)

    def test_chunk_docs(self):
        """
        Test the chunk_docs function.

        This method tests the chunk_docs function with different rag types and verifies the expected outputs.

        Returns:
            None
        """
        # Test case 1: Standard rag
        print('Testing standard rag...')
        result = chunk_docs(self.docs, 
                            rag_type=self.rag_type['Standard'], 
                            chunk_method=self.chunk_method, 
                            chunk_size=self.chunk_size, 
                            chunk_overlap=self.chunk_overlap)
        self.assertEqual(result['rag'], self.rag_type['Standard'])
        self.assertIsNotNone(result['pages'])
        self.assertIsNotNone(result['chunks'])
        self.assertIsNotNone(result['splitters'])
        print('Standard rag test passed!')

        # Test case 2: Parent-Child rag
        print('Testing parent-child rag...')
        result = chunk_docs(self.docs, 
                            rag_type=self.rag_type['Parent-Child'], 
                            chunk_method=self.chunk_method, 
                            chunk_size=self.chunk_size, 
                            chunk_overlap=self.chunk_overlap)
        self.assertEqual(result['rag'], self.rag_type['Parent-Child'])
        self.assertIsNotNone(result['pages']['doc_ids'])
        self.assertIsNotNone(result['pages']['parent_chunks'])
        self.assertIsNotNone(result['chunks'])
        self.assertIsNotNone(result['splitters']['parent_splitter'])
        self.assertIsNotNone(result['splitters']['child_splitter'])
        print('Parent-child rag test passed!')

        # Test case 3: Summary rag
        print('Testing summary rag...')
        result = chunk_docs(self.docs, 
                            rag_type=self.rag_type['Summary'], 
                            chunk_method=self.chunk_method, 
                            chunk_size=self.chunk_size, 
                            chunk_overlap=self.chunk_overlap, 
                            llm=self.llm['Hugging Face'])
        self.assertEqual(result['rag'], self.rag_type['Summary'])
        self.assertIsNotNone(result['pages']['doc_ids'])
        self.assertIsNotNone(result['pages']['docs'])
        self.assertIsNotNone(result['summaries'])
        self.assertEqual(result['llm'], self.llm['Hugging Face'])
        print('Summary rag test passed!')
    
    def generate_test_cases(self, export:bool=False):
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
        # TODO: throw in bad inputs for each of the 4 major types below.
        test_cases = [
            {
                # Tests ChromaDB setups, advanced RAG (standard/parent-child)
                'index_type': [self.index_type['ChromaDB']],
                'query_model': [self.query_model['OpenAI']],
                'rag_type': [self.rag_type['Standard'], self.rag_type['Parent-Child']],
                'llm': [self.llm['Hugging Face']]
            },
            {
                # Tests advanced RAG (summary) and LLM (openai/hugging face)
                'index_type': [self.index_type['ChromaDB']],
                'query_model': [self.query_model['OpenAI']],
                'rag_type': [self.rag_type['Summary']],
                'llm': [self.llm['OpenAI'], self.llm['Hugging Face']]
            },
            {
                # Tests Pinecone setups, embedding types (openai/voyage)
                'index_type': [self.index_type['Pinecone']],
                'query_model': [self.query_model['OpenAI'], self.query_model['Voyage']],
                'rag_type': [self.rag_type['Standard']],
                'llm': [self.llm['Hugging Face']]
            },
            {
                # Tests RAGatouille setup
                'index_type': [self.index_type['RAGatouille']],
                'query_model': [self.query_model['RAGatouille']],
                'rag_type': [self.rag_type['Standard']],
                'llm': [self.llm['Hugging Face']]
            }
        ]
        tests = permute_tests(test_cases)
        
        if export:
            file_path = 'test_cases.json'
            with open(file_path, 'w') as json_file:
                json.dump([{**test, 'id': i+1} for i, test in enumerate(tests)], json_file, default=custom_encoder, indent=4)
        
        return tests

    def test_process(self):
            """
            Tests the entire process of initializing a database, upserting documents, and querying the database.

            Returns:
                None
            """
            tests=self.generate_test_cases(export=True)
            # Run the tests as subtests
            for i, test in enumerate(tests):
                with self.subTest(i=i):
                    print(f'Running test {i+1}...')
                    try:
                        chunker = chunk_docs(self.docs, 
                                            rag_type=test['rag_type'], 
                                            chunk_method=self.chunk_method, 
                                            chunk_size=self.chunk_size, 
                                            chunk_overlap=self.chunk_overlap,
                                            llm=test['llm'])
                        self.assertIsNotNone(chunker)
                        print('Docs chunked!')
                        
                        vectorstore=initialize_database(test['index_type'], 
                                            index_name='test', 
                                            query_model=test['query_model'],
                                            rag_type=test['rag_type'],
                                            local_db_path=self.LOCAL_DB_PATH, 
                                            clear=True,
                                            test_query=False,
                                            init_ragatouille=True)
                        self.assertIsNotNone(vectorstore)
                        print('Database initialized!')
                        
                        vectorstore, retriever = upsert_docs(test['index_type'], 
                                            'test', 
                                            vectorstore,
                                            chunker,
                                            batch_size=100, 
                                            local_db_path=self.LOCAL_DB_PATH)
                        self.assertIsNotNone(vectorstore)
                        self.assertIsNotNone(retriever)
                        print('Docs upserted!')
                        
                        # Simulate reloading in the RAGatouille index.
                        if test['index_type']=='RAGatouille':
                            query_model_qa=RAGPretrainedModel.from_index(self.LOCAL_DB_PATH+'/.ragatouille/colbert/indexes/'+'test')
                        else:
                            query_model_qa=test['query_model']
                        qa_model_obj=QA_Model(test['index_type'],
                                        'test',
                                        query_model_qa,
                                        test['llm'],
                                        rag_type=test['rag_type'],
                                        test_query=False,
                                        local_db_path=self.LOCAL_DB_PATH)
                        # Object setup
                        self.assertIsNotNone(qa_model_obj)
                        qa_model_obj.query_docs(self.test_prompt)
                        # Queryz
                        response=qa_model_obj.ai_response
                        self.assertIsNotNone(response)
                        # Alternate question
                        alternate_question=qa_model_obj.generate_alternative_questions(self.test_prompt,response=response)
                        self.assertIsNotNone(alternate_question)
                        #Show results
                        print(f'Query response: {response}')
                        print(f'Alternate questions: {alternate_question}')
                        print('Query and alternative question successful!')

                        delete_index(test['index_type'],
                                     'test',
                                     test['rag_type'],
                                     local_db_path=self.LOCAL_DB_PATH)
                        print('Database deleted!')
                    except Exception as e:
                        delete_index(test['index_type'],
                                     'test',
                                     test['rag_type'],
                                     local_db_path=self.LOCAL_DB_PATH)
                        raise e # Pass on exception after delete index is used.
if __name__ == '__main__':
    unittest.main()