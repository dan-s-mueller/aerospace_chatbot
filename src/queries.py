import os
import logging
import re

from dotenv import load_dotenv, find_dotenv

import openai
import pinecone
from pinecone import Pinecone as pinecone_client
import chromadb

from langchain_pinecone import Pinecone
from langchain_community.vectorstores import Chroma

from langchain.memory import ConversationBufferMemory

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string

from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, DEFAULT_DOCUMENT_PROMPT, TEST_QUERY_PROMPT

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

# Class and functions
class QA_Model:
    def __init__(self, 
                 index_type,
                 index_name,
                 query_model,
                 llm,
                 k=6,
                 search_type='similarity',
                 fetch_k=50,
                 temperature=0,
                 chain_type='stuff',
                 filter_arg=False,
                 local_db_path='../db'):
        
        self.index_type=index_type
        self.index_name=index_name
        self.query_model=query_model
        self.llm=llm
        self.k=k
        self.search_type=search_type
        self.fetch_k=fetch_k
        self.temperature=temperature
        self.chain_type=chain_type
        self.filter_arg=filter_arg
        self.sources=[]

        load_dotenv(find_dotenv(),override=True)

        # Define retriever search parameters
        search_kwargs = _process_retriever_args(self.filter_arg,
                                                self.sources,
                                                self.search_type,
                                                self.k,
                                                self.fetch_k)

        # Read in from the vector database
        if index_type=='Pinecone':
            pc = pinecone_client(api_key=PINECONE_API_KEY)
            logging.info('Chat pinecone index name: '+str(index_name))
            logging.info('Chat query model: '+str(query_model))
            try:
                pc.describe_index(index_name)
            except:
                raise ValueError(f"Index {index_name} does not exist. Please create it first.")
            else:
                logging.info(f"Index {index_name} exists.")
            index = pc.Index(index_name)
            self.vectorstore = Pinecone(index,query_model,'page_content')
            logging.info('Chat vectorstore: '+str(self.vectorstore))

            # Test query
            try:
                test_query = self.vectorstore.similarity_search(TEST_QUERY_PROMPT)
            except:
                raise Exception("Pinecone vector database is not configured properly. Test query failed. Likely the index does not exist.")
            logging.info('Test query: '+str(test_query))
            if not test_query:
                raise ValueError("Pinecone vector database is not configured properly. Test query failed.")
            else:
                logging.info('Test query succeeded!')
            
            self.retriever=self.vectorstore.as_retriever(search_type=search_type,
                                                         search_kwargs=search_kwargs)
            logging.info('Chat retriever: '+str(self.retriever))
        elif index_type=='ChromaDB':
            logging.info('Chat chroma index name: '+str(index_name))
            logging.info('Chat query model: '+str(query_model))
            logging.info('Local db path: '+str(local_db_path))
            try:
                persistent_client = chromadb.PersistentClient(path=local_db_path+'/chromadb')   
            except:
                raise ValueError("Chroma vector database needs to be reset. Clear cache.")
            self.vectorstore = Chroma(client=persistent_client,
                                      collection_name=index_name,
                                      embedding_function=query_model)
            logging.info('Chat vectorstore: '+str(self.vectorstore))

            # Test query
            try:
                test_query = self.vectorstore.similarity_search(TEST_QUERY_PROMPT)
            except:
                raise Exception("Chroma vector database is not configured properly. Test query failed. Likely the index does not exist.")
            logging.info('Test query: '+str(test_query))
            if not test_query:
                raise ValueError("Chroma vector database or llm is not configured properly. Test query failed.")
            else:
                logging.info('Test query succeeded!')
            
            self.retriever=self.vectorstore.as_retriever(search_type=search_type,
                                                         search_kwargs=search_kwargs)
            logging.info('Chat retriever: '+str(self.retriever))
        elif index_type=='RAGatouille':
            # Easy because the index is picked up directly.
            self.vectorstore=query_model
            logging.info('Chat query model:'+str(query_model))

             # Test query
            try:
                test_query = self.vectorstore.search(TEST_QUERY_PROMPT)
            except:
                raise Exception("RAGatouille vector database is not configured properly.")
            logging.info('Test query: '+str(test_query))
            if not test_query:
                raise ValueError("Chroma vector database is not configured properly. Test query failed.")
            else:
                logging.info('Test query succeeded!')
            
            self.retriever=self.vectorstore.as_langchain_retriever()
            logging.info('Chat retriever: '+str(self.retriever))

        # Intialize memory
        self.memory = ConversationBufferMemory(
                        return_messages=True, output_key='answer', input_key='question')
        logging.info('Memory: '+str(self.memory))

        # Assemble main chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.retriever,
                                                      self.memory,
                                                      self.search_type,
                                                      search_kwargs)
    def query_docs(self,query):        
        self.memory.load_memory_variables({})
        logging.info('Memory content before qa result: '+str(self.memory))

        logging.info('Query: '+str(query))
        self.result = self.conversational_qa_chain.invoke({'question': query})
        logging.info('QA result: '+str(self.result))

        if self.index_type!='RAGatouille':
            self.sources = '\n'.join(str(data.metadata) for data in self.result['references'])
            if self.llm.__class__.__name__=='ChatOpenAI':
                self.ai_response=self.result['answer'].content + '\nSources: \n'+self.sources
            elif self.llm.__class__.__name__=='HuggingFaceHub':
                self.ai_response=self.result['answer'] + '\nSources: \n'+self.sources
            logging.info('Sources: '+str(self.sources))
            logging.info('Response with sources: '+str(self.ai_response))
        else:
            # RAGatouille doesn't have metadata, need to extract from context first.
            extracted_metadata = []
            pattern = r'\{([^}]*)\}(?=[^{}]*$)' # Regular expression pattern to match the last curly braces

            for ref in self.result['references']:
                match = re.search(pattern, ref.page_content)
                if match:
                    extracted_metadata.append("{"+match.group(1)+"}")
            self.sources = '\n'.join(extracted_metadata)

            if self.llm.__class__.__name__=='ChatOpenAI':
                self.ai_response=self.result['answer'].content + '\nSources: \n'+self.sources
            elif self.llm.__class__.__name__=='HuggingFaceHub':
                self.ai_response=self.result['answer'] + '\nSources: \n'+self.sources
            logging.info('Sources: '+str(self.sources))
            logging.info('Response with sources: '+str(self.ai_response))

        self.memory.save_context({'question': query}, {'answer': self.ai_response})
        logging.info('Memory content after qa result: '+str(self.memory))

    def update_model(self,
                     llm,
                     k=6,
                     search_type='similarity',
                     fetch_k=50,
                     filter_arg=False):

        self.llm=llm
        self.k=k
        self.search_type=search_type
        self.fetch_k=fetch_k
        self.filter_arg=filter_arg
        
        # Define retriever search parameters
        search_kwargs = _process_retriever_args(self.filter_arg,
                                                self.sources,
                                                self.search_type,
                                                self.k,
                                                self.fetch_k)
        # Update conversational retrieval chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.retriever,
                                                      self.memory,
                                                      self.search_type,
                                                      search_kwargs)
        logging.info('Updated qa chain: '+str(self.conversational_qa_chain))

# Internal functions
def _combine_documents(docs, 
                        document_prompt=DEFAULT_DOCUMENT_PROMPT, 
                        document_separator='\n\n'):
    '''
    Combine a list of documents into a single string.
    '''
    # TODO: this would be where stuff, map reduce, etc. would go
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
def _define_qa_chain(llm,
                     retriever,
                     memory,
                     search_type,
                     search_kwargs):
    '''
    Define the conversational QA chain.
    '''
    # This adds a 'memory' key to the input object
    loaded_memory = RunnablePassthrough.assign(
                        chat_history=RunnableLambda(memory.load_memory_variables) 
                        | itemgetter('history'))  
    logging.info('Loaded memory: '+str(loaded_memory))
    
    # Assemble main chain
    standalone_question = {
        'standalone_question': {
            'question': lambda x: x['question'],
            'chat_history': lambda x: get_buffer_string(x['chat_history'])}
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()}
    logging.info('Condense inputs as a standalong question: '+str(standalone_question))
    retrieved_documents = {
        'source_documents': itemgetter('standalone_question') 
                            | retriever,
        'question': lambda x: x['standalone_question']}
    logging.info('Retrieved documents: '+str(retrieved_documents))
    # Now we construct the inputs for the final prompt
    final_inputs = {
        'context': lambda x: _combine_documents(x['source_documents']),
        'question': itemgetter('question')}
    logging.info('Combined documents: '+str(final_inputs))
    # And finally, we do the part that returns the answers
    answer = {
        'answer': final_inputs 
                    | QA_PROMPT 
                    | llm,
        'references': itemgetter('source_documents')}
    conversational_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer
    logging.info('Conversational QA chain: '+str(conversational_qa_chain))
    return conversational_qa_chain
def _process_retriever_args(filter_arg,
                            sources,
                            search_type,
                            k,
                            fetch_k):
    '''
    Process arguments for retriever.
    '''
    # Implement filter
    if filter_arg:
        filter_list = list(set(item['source'] for item in sources[-1]))
        filter_items=[]
        for item in filter_list:
            filter_item={'source': item}
            filter_items.append(filter_item)
        filter={'$or':filter_items}
    else:
        filter=None

    # Impement filtering and number of documents to return
    if search_type=='mmr':
        search_kwargs={'k':k,'fetch_k':fetch_k,'filter':filter} # See as_retriever docs for parameters
    else:
        search_kwargs={'k':k,'filter':filter} # See as_retriever docs for parameters
    
    return search_kwargs