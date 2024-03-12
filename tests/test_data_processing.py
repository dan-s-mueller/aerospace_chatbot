import unittest
import os
import sys
import logging
import json
import itertools
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from ragatouille import RAGPretrainedModel
from langchain_openai import ChatOpenAI

sys.path.append('../src')
from data_processing import chunk_docs, initialize_database, upsert_docs, delete_index
from queries import QA_Model

logging.basicConfig(filename='test_data_processing.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
load_dotenv(find_dotenv(),override=True)

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
LOCAL_DB_PATH=os.getenv('LOCAL_DB_PATH')

def permute_tests(test_data):
    """
    Permute test data to generate all possible combinations.
    Data is in a list of dicts, where each has keys to be iterated. 
    Example: [{'index_type': ['ChromaDB'], 'rag_type': ['Standard', 'Parent-Child', 'Summary']}, 
              {'index_type': [Pinecone], 'rag_type': ['Standard', 'Parent-Child']}]
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
    """
    if hasattr(obj, '__str__'):
        return str(obj)
    # Insert more custom handling here if necessary
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Fixed inputs
        self.docs = ['test1.pdf', 'test2.pdf']
        self.chunk_method='character_recursive'
        self.chunk_size=500
        self.chunk_overlap=0
        self.test_prompt='What are some nuances associated with the analysis and design of hinged booms?'   # Info on test2.pdf

        # Variable inputs
        self.llm={'OpenAI':ChatOpenAI(model_name='gpt-3.5-turbo-1106', # Openai
                                      openai_api_key=OPENAI_API_KEY,
                                      max_tokens=500), 
                  'Hugging Face':ChatOpenAI(base_url='https://api-inference.huggingface.co/v1',  # Hugging face
                                            model='mistralai/Mistral-7B-Instruct-v0.2',
                                            api_key=HUGGINGFACEHUB_API_TOKEN,
                                            max_tokens=500)}
        self.query_model={'OpenAI':OpenAIEmbeddings(model='text-embedding-ada-002',openai_api_key=OPENAI_API_KEY),
                          'Voyage':VoyageEmbeddings(voyage_api_key=VOYAGE_API_KEY),
                          'RAGatouille':'colbert-ir/colbertv2.0'}
        self.index_type = {index: index for index in ['ChromaDB', 'Pinecone', 'RAGatouille']}
        self.rag_type = {rag: rag for rag in ['Standard','Parent-Child','Summary']}

    def test_chunk_docs(self):
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
    
    def test_process(self):
        """
        Tests the entire process of initializing a database, upserting documents, and querying the database.
        """
        # Determine the set of cases to screen
        test_cases=[{
             # Tests ChromaDB setups, advanced RAG (standard/parent-child)
            'index_type': [self.index_type['ChromaDB']],
            'query_model': [self.query_model['OpenAI']],
            'rag_type': [self.rag_type['Standard'],self.rag_type['Parent-Child']],
            'llm': [self.llm['Hugging Face']]
        },
        {
             # Tests advanced RAG (summary) and LLM (openai/hugging face)
            'index_type': [self.index_type['ChromaDB']],
            'query_model': [self.query_model['OpenAI']],
            'rag_type': [self.rag_type['Summary']],
            'llm': [self.llm['OpenAI'],self.llm['Hugging Face']]
        },
        {
            # Tests Pinecone setups, embedding types (openai/voyage)
            'index_type': [self.index_type['Pinecone']],
            'query_model': [self.query_model['OpenAI'],self.query_model['Voyage']],
            'rag_type': [self.rag_type['Standard']],
            'llm': [self.llm['Hugging Face']]
        },
        {
            # Tests RAGatouille setup
            'index_type': [self.index_type['RAGatouille']],
            'query_model': [self.query_model['RAGatouille']],
            'rag_type': [self.rag_type['Standard']],
            'llm': [self.llm['Hugging Face']]
        }]
        tests=permute_tests(test_cases)
        file_path = 'test_cases.json'
        with open(file_path, 'w') as json_file:
            json.dump([{**test, 'id': i+1} for i, test in enumerate(tests)], json_file, default=custom_encoder, indent=4)

        # Run the tests as subtests
        for i, test in enumerate(tests):
            with self.subTest(i=i):
                print(f'Running test {i+1}...')
                
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
                                    local_db_path=LOCAL_DB_PATH, 
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
                                    local_db_path=LOCAL_DB_PATH)
                self.assertIsNotNone(vectorstore)
                self.assertIsNotNone(retriever)
                print('Docs upserted!')
                
                qa_model_obj=QA_Model(test['index_type'],
                                'test',
                                 test['query_model'],
                                 test['llm'],
                                 rag_type=test['rag_type'],
                                 local_db_path=LOCAL_DB_PATH)
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
if __name__ == '__main__':
    unittest.main()