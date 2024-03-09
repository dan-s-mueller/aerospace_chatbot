import data_processing

import os
import logging
import re
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

import openai
import pinecone
from pinecone import Pinecone as pinecone_client
import chromadb

from langchain_pinecone import Pinecone
from langchain_community.vectorstores import Chroma

from langchain.memory import ConversationBufferMemory

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

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
    """A class representing a Question-Answering Model.

    Args:
        index_type (str): The type of index.
        index_name (str): The name of the index.
        query_model (str): The query model.
        llm (str): The language model.
        rag_type (str, optional): The type of RAG model. Defaults to 'Standard'.
        k (int, optional): The number of retriever results. Defaults to 6.
        search_type (str, optional): The type of search. Defaults to 'similarity'.
        fetch_k (int, optional): The number of documents to fetch. Defaults to 50.
        temperature (int, optional): The temperature for generation. Defaults to 0.
        chain_type (str, optional): The type of chain. Defaults to 'stuff'.
        filter_arg (bool, optional): Whether to filter arguments. Defaults to False.
        local_db_path (str, optional): The local database path. Defaults to '../db'.

    Attributes:
        index_type (str): The type of index.
        index_name (str): The name of the index.
        query_model (str): The query model.
        llm (str): The language model.
        rag_type (str): The type of RAG model.
        k (int): The number of retriever results.
        search_type (str): The type of search.
        fetch_k (int): The number of documents to fetch.
        temperature (int): The temperature for generation.
        chain_type (str): The type of chain.
        filter_arg (bool): Whether to filter arguments.
        local_db_path (str): The local database path.
        sources (list): The list of sources.
        vectorstore (VectorStore): The vector store.
        retriever (Retriever): The retriever.
        memory (ConversationBufferMemory): The conversation buffer memory.
        conversational_qa_chain (ConversationalQAChain): The conversational QA chain.
    """
    def __init__(self, 
                 index_type,
                 index_name,
                 query_model,
                 llm,
                 rag_type='Standard',
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
        self.rag_type=rag_type
        self.k=k
        self.search_type=search_type
        self.fetch_k=fetch_k
        self.temperature=temperature
        self.chain_type=chain_type
        self.filter_arg=filter_arg
        self.local_db_path=local_db_path
        self.sources=[]

        load_dotenv(find_dotenv(),override=True)

        # Define retriever search parameters
        search_kwargs = _process_retriever_args(self.filter_arg,
                                                self.sources,
                                                self.search_type,
                                                self.k,
                                                self.fetch_k)

        # Read in from the vector database
        self.vectorstore=data_processing.initialize_database(self.index_type,
                                                             self.index_name,
                                                             self.query_model,
                                                             self.rag_type,
                                                             local_db_path=self.local_db_path,
                                                             test_query=True,
                                                             init_ragatouille=False)  
        if self.rag_type=='Standard':  
            self.retriever=self.vectorstore.as_retriever(search_type=search_type,
                                                        search_kwargs=search_kwargs)
        elif self.rag_type=='Parent-Child':
            lfs = LocalFileStore(Path(self.local_db_path).resolve() / 'local_file_store' / index_name)
            self.retriever = MultiVectorRetriever(
                                vectorstore=self.vectorstore,
                                byte_store=lfs,
                                id_key="doc_id",
                            )
        elif self.rag_type=='Summary' or 'Multi-Query':
            raise NotImplementedError
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
        """
        Executes a query and retrieves the relevant documents.

        Args:
            query (str): The query string.

        Returns:
            None
        """       
        self.memory.load_memory_variables({})
        logging.info('Memory content before qa result: '+str(self.memory))

        logging.info('Query: '+str(query))
        self.result = self.conversational_qa_chain.invoke({'question': query})
        logging.info('QA result: '+str(self.result))

        if self.index_type!='RAGatouille':
            self.sources = '\n'.join(str(data.metadata) for data in self.result['references'])
            if self.llm.__class__.__name__=='ChatOpenAI':
                self.ai_response = self.result['answer'].content + '\n\nSources:\n' + self.sources
            else:
                raise NotImplementedError
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
                self.ai_response=self.result['answer'].content + '\n\nSources:\n' + self.sources
            else:
                raise NotImplementedError
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
        """
        Updates the model with new parameters.

        Args:
            llm (object): The language model to be used for retrieval.
            k (int, optional): The number of retriever candidates to consider. Defaults to 6.
            search_type (str, optional): The type of search to perform. Defaults to 'similarity'.
            fetch_k (int, optional): The number of documents to fetch from the retriever. Defaults to 50.
            filter_arg (bool, optional): Whether to apply additional filtering during retrieval. Defaults to False.
        """

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
    # TODO: this would be where stuff, map reduce, etc. would go
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
def _define_qa_chain(llm,
                     retriever,
                     memory,
                     search_type,
                     search_kwargs):
    """
    Defines the conversational QA chain for the chatbot. Based on this: https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain

    Args:
        llm: The language model component.
        retriever: The document retriever component.
        memory: The memory component.
        search_type: The type of search to be performed.
        search_kwargs: Additional keyword arguments for the search.

    Returns:
        The conversational QA chain.

    """
    
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
    """
    Process the arguments for the retriever function.

    Args:
        filter_arg (bool): Whether to apply filtering or not.
        sources (list): List of sources.
        search_type (str): Type of search.
        k (int): Number of documents to retrieve.
        fetch_k (int): Number of documents to fetch.

    Returns:
        dict: Dictionary containing the processed search arguments.

    """
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

    # Implement filtering and number of documents to return
    if search_type=='mmr':
        search_kwargs={'k':k,'fetch_k':fetch_k,'filter':filter} # See as_retriever docs for parameters
    else:
        search_kwargs={'k':k,'filter':filter} # See as_retriever docs for parameters
    
    return search_kwargs