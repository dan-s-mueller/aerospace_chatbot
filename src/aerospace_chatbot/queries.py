import data_processing

import os
import logging
import re
from pathlib import Path

import openai
import pinecone
from pinecone import Pinecone as pinecone_client
import chromadb

from langchain_openai import ChatOpenAI

from langchain_pinecone import Pinecone
from langchain_community.vectorstores import Chroma

from langchain.memory import ConversationBufferMemory

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string

from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, DEFAULT_DOCUMENT_PROMPT, GENERATE_SIMILAR_QUESTIONS, GENERATE_SIMILAR_QUESTIONS_W_CONTEXT

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

# Class and functions
class QA_Model:
    """
    Represents a Question-Answering Model.

    Args:
        index_type (str): The type of index.
        index_name (str): The name of the index.
        query_model (str): The query model.
        llm (ChatOpenAI): The language model for generating responses.
        rag_type (str, optional): The type of RAG model. Defaults to 'Standard'.
        k (int, optional): The number of retriever results to consider. Defaults to 6.
        search_type (str, optional): The type of search to perform. Defaults to 'similarity'.
        fetch_k (int, optional): The number of documents to fetch from the retriever. Defaults to 50.
        temperature (int, optional): The temperature for response generation. Defaults to 0.
        local_db_path (str, optional): The path to the local database. Defaults to '.'.

    Attributes:
        index_type (str): The type of index.
        index_name (str): The name of the index.
        query_model (str): The query model.
        llm (ChatOpenAI): The language model for generating responses.
        rag_type (str): The type of RAG model.
        k (int): The number of retriever results to consider.
        search_type (str): The type of search to perform.
        fetch_k (int): The number of documents to fetch from the retriever.
        temperature (int): The temperature for response generation.
        local_db_path (str): The path to the local database.
        sources (list): The list of sources for the retrieved documents.
        vectorstore (VectorStore): The vector store for document retrieval.
        retriever (Retriever): The retriever for document retrieval.
        memory (ConversationBufferMemory): The memory for conversation history.
        conversational_qa_chain (Chain): The chain for conversational QA.

    """
    def __init__(self, 
                 index_type,
                 index_name,
                 query_model,
                 llm:ChatOpenAI,
                 rag_type='Standard',
                 k=6,
                 search_type='similarity',
                 fetch_k=50,
                 temperature=0,
                 local_db_path='.'):
        """
        Initializes a new instance of the QA_Model class.

        Args:
            index_type (str): The type of index.
            index_name (str): The name of the index.
            query_model (str): The query model.
            llm (ChatOpenAI): The language model for generating responses.
            rag_type (str, optional): The type of RAG model. Defaults to 'Standard'.
            k (int, optional): The number of retriever results to consider. Defaults to 6.
            search_type (str, optional): The type of search to perform. Defaults to 'similarity'.
            fetch_k (int, optional): The number of documents to fetch from the retriever. Defaults to 50.
            temperature (int, optional): The temperature for response generation. Defaults to 0.
            local_db_path (str, optional): The path to the local database. Defaults to '.'.

        """
        self.index_type=index_type
        self.index_name=index_name
        self.query_model=query_model
        self.llm=llm
        self.rag_type=rag_type
        self.k=k
        self.search_type=search_type
        self.fetch_k=fetch_k
        self.temperature=temperature
        self.local_db_path=local_db_path
        self.sources=[]

        # Define retriever search parameters
        search_kwargs = _process_retriever_args(self.search_type,
                                                self.k,
                                                self.fetch_k)

        # Read in from the vector database
        self.vectorstore=data_processing.initialize_database(self.index_type,
                                                             self.index_name,
                                                             self.query_model,
                                                             self.rag_type,
                                                             local_db_path=self.local_db_path,
                                                             init_ragatouille=False)  
        if self.rag_type=='Standard':  
            if self.index_type=='ChromaDB' or self.index_type=='Pinecone':
                self.retriever=self.vectorstore.as_retriever(search_type=self.search_type)
            elif self.index_type=='RAGatouille':
                self.retriever=self.vectorstore.as_langchain_retriever()
        elif self.rag_type=='Parent-Child' or self.rag_type=='Summary':
            self.lfs = LocalFileStore(Path(self.local_db_path).resolve() / 'local_file_store' / self.index_name)
            self.retriever = MultiVectorRetriever(
                                vectorstore=self.vectorstore,
                                byte_store=self.lfs,
                                id_key="doc_id",
                            )
        else:
            raise NotImplementedError

        # Intialize memory
        self.memory = ConversationBufferMemory(
                        return_messages=True, output_key='answer', input_key='question')

        # Assemble main chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.retriever,
                                                      self.memory,
                                                      kwargs=search_kwargs)
    def query_docs(self,query): 
        """
        Executes a query and retrieves the relevant documents.

        Args:
            query (str): The query string.

        Returns:
            None
        """       
        self.memory.load_memory_variables({})
        self.result = self.conversational_qa_chain.invoke({'question': query})

        if self.index_type!='RAGatouille':
            self.sources = '\n'.join(str(data.metadata) for data in self.result['references'])
            if self.llm.__class__.__name__=='ChatOpenAI':
                self.ai_response = self.result['answer'].content + '\n\nSources:\n' + self.sources
            else:
                raise NotImplementedError
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

        self.memory.save_context({'question': query}, {'answer': self.ai_response})
    def update_model(self,
                     llm:ChatOpenAI,
                     search_type='similarity',
                     k=6,
                     fetch_k=50):
        """
        Updates the model with new parameters.

        Args:
            llm (ChatOpenAI): The language model for generating responses.
            search_type (str, optional): The type of search to perform. Defaults to 'similarity'.
            k (int, optional): The number of retriever results to consider. Defaults to 6.
            fetch_k (int, optional): The number of documents to fetch from the retriever. Defaults to 50.

        Returns:
            None
        """
        self.llm=llm
        self.search_type=search_type
        self.k=k
        self.fetch_k=fetch_k

        # Define retriever search parameters
        search_kwargs = _process_retriever_args(search_type=self.search_type,
                                                k=self.k,
                                                fetch_k=self.fetch_k)
        
        # Update conversational retrieval chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.retriever,
                                                      self.memory,
                                                      kwargs=search_kwargs)
        logging.info('Updated qa chain: '+str(self.conversational_qa_chain))
    def generate_alternative_questions(self,
                                       prompt:str,
                                       response=None):
        """
        Generates alternative questions based on a prompt.

        Args:
            prompt (str): The prompt for generating alternative questions.
            response (str, optional): The response context. Defaults to None.

        Returns:
            str: The generated alternative questions.
        """
        if response:
            prompt_template=GENERATE_SIMILAR_QUESTIONS_W_CONTEXT
            invoke_dict={'question':prompt,'context':response}
        else:
            prompt_template=GENERATE_SIMILAR_QUESTIONS
            invoke_dict={'question':prompt}
        
        chain = (
                prompt_template
                | self.llm
                | StrOutputParser()
            )
        
        alternative_questions=chain.invoke(invoke_dict)
        logging.info('Generated alternative questions: '+str(alternative_questions))
        
        return alternative_questions

# Internal functions
def _combine_documents(docs, 
                        document_prompt=DEFAULT_DOCUMENT_PROMPT, 
                        document_separator='\n\n'):
    """Combines a list of documents into a single string.

    Args:
        docs (list): A list of documents to be combined.
        document_prompt (str, optional): The prompt to be added before each document. Defaults to DEFAULT_DOCUMENT_PROMPT.
        document_separator (str, optional): The separator to be added between each document. Defaults to '\n\n'.

    Returns:
        str: The combined string of all the documents.
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
def _define_qa_chain(llm,
                     retriever,
                     memory,
                     kwargs=None):
    """Defines the conversational QA chain.

    Args:
        llm: The language model component.
        retriever: The document retriever component.
        memory: The memory component.
        kwargs: Optional keyword arguments.

    Returns:
        The conversational QA chain.

    Raises:
        None.
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
def _process_retriever_args(search_type='similarity',
                            k=6,
                            fetch_k=50):
    """
    Process the retriever arguments.

    Args:
        search_type (str, optional): The type of search. Defaults to 'similarity'.
        k (int, optional): The number of documents to retrieve. Defaults to 6.
        fetch_k (int, optional): The number of documents to fetch. Defaults to 50.

    Returns:
        dict: The search arguments for the retriever.
    """
    # TODO add functionality for filter if required
    # # Implement filter
    # if filter_arg:
    #     filter_list = list(set(item['source'] for item in sources[-1]))
    #     filter_items=[]
    #     for item in filter_list:
    #         filter_item={'source': item}
    #         filter_items.append(filter_item)
    #     filter={'$or':filter_items}
    # else:
    #     filter=None

    # Implement filtering and number of documents to return
    if search_type=='mmr':
        search_kwargs={'k':k,'fetch_k':fetch_k,'filter':filter} # See as_retriever docs for parameters
    else:
        search_kwargs={'k':k,'filter':filter} # See as_retriever docs for parameters
    
    return search_kwargs