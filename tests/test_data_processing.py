import unittest
import os
import sys
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from ragatouille import RAGPretrainedModel
from langchain_openai import ChatOpenAI

sys.path.append('../src')
from data_processing import chunk_docs, initialize_database, upsert_docs, delete_index

load_dotenv(find_dotenv(),override=True)

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
LOCAL_DB_PATH=os.getenv('LOCAL_DB_PATH')

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Fixed inputs
        self.docs = ['test1.pdf', 'test2.pdf']
        self.chunk_method='character_recursive'
        self.chunk_size=500
        self.chunk_overlap=0

        # Variable inputs
        self.llm={'OpenAI':ChatOpenAI(model_name='gpt-3.5-turbo-1106', # Openai
                                openai_api_key=OPENAI_API_KEY), 
                  'Hugging Face':ChatOpenAI(base_url='https://api-inference.huggingface.co/v1',  # Hugging face
                                            model='mistralai/Mistral-7B-Instruct-v0.2',
                                            api_key=HUGGINGFACEHUB_API_TOKEN)}
        self.query_model={'OpenAI':OpenAIEmbeddings(model='text-embedding-ada-002',openai_api_key=OPENAI_API_KEY),
                          'Voyage':VoyageEmbeddings(voyage_api_key=VOYAGE_API_KEY)}
        self.index_type = {index: index for index in ['ChromaDB', 'Pinecone', 'Ragatouille']}
        self.rag_type = {rag: rag for rag in ['Standard','Parent-Child','Summary']}

    def test_chunk_docs(self):
        # Test case 1: Standard rag
        print('Testing standard rag...')
        result = chunk_docs(self.docs, 
                            rag_type=self.rag_type[0], 
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
        self.assertEqual(result['rag'], self.rag_type[1])
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
                            llm=self.llm)
        self.assertEqual(result['rag'], self.rag_type[2])
        self.assertIsNotNone(result['pages']['doc_ids'])
        self.assertIsNotNone(result['pages']['docs'])
        self.assertIsNotNone(result['summaries'])
        self.assertEqual(result['llm'], self.llm)
        print('Summary rag test passed!')
    
    def test_inialize_upsert_delete_database(self):
        # Test 1: ChromaDB + OpenAI + Standard RAG
        initialize_database(self.index_type['ChromaDB'], 
                            index_name='test', 
                            query_model=self.query_model['OpenAI'],
                            rag_type=self.rag_type['Standard'],
                            local_db_path=LOCAL_DB_PATH, 
                            clear=True,
                            test_query=True,
                            init_ragatouille=True)
    

        # Test 2: Pinecone
        # Test 3: Ragatouille

if __name__ == '__main__':
    unittest.main()