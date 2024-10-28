import os
from pathlib import Path
from typing import Union

import openai
import pinecone
from pinecone import Pinecone as pinecone_client
import chromadb

from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import Pinecone
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferMemory

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string

import data_processing
from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, DEFAULT_DOCUMENT_PROMPT, GENERATE_SIMILAR_QUESTIONS, GENERATE_SIMILAR_QUESTIONS_W_CONTEXT

# Class and functions
class QA_Model:
    """
    Represents a Question-Answering Model.
    """
    def __init__(self, 
                 index_type:str,
                 index_name:str,
                 query_model:object,
                 llm:Union[ChatAnthropic, ChatOpenAI],
                 rag_type:str='Standard',
                 k:int=6,
                 search_type:str='similarity',
                 fetch_k:int=50,
                 temperature:int=0,
                 local_db_path:str='.',
                 user_doc_namespace:str=None,
                 reset_query_db:bool=False):
        """
        Initializes a new instance of the QA_Model class.

        Args:
            index_type (str): The type of document index.
            index_name (str): The name of the document index.
            query_model (object): The query model.
            llm (ChatAnthropic or ChatOpenAI): The language model for generating responses.
            rag_type (str, optional): The type of RAG model.
            k (int, optional): The number of retriever results to consider.
            search_type (str, optional): The type of search to perform.
            fetch_k (int, optional): The number of documents to fetch from the retriever.
            temperature (int, optional): The temperature for response generation.
            local_db_path (str, optional): The path to the local database.
            user_doc_namespace (str, optional): The namespace for user documents.
            reset_query_db (bool, optional): Whether to reset the query database.
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
        self.user_doc_namespace=user_doc_namespace

        self.memory=None
        self.result=None
        self.sources=None
        self.ai_response=None
        self.conversational_qa_chain=None

        self.doc_vectorstore=None
        self.query_vectorstore=None
        self.retriever=None

        # Read in from the vector database
        self.doc_vectorstore=data_processing.initialize_database(self.index_type,
                                                                 self.index_name,
                                                                 self.query_model,
                                                                 self.rag_type,
                                                                 local_db_path=self.local_db_path,
                                                                 init_ragatouille=False)
        
        # Iniialize a database to capture queries in a temp database. If an existing database exists with questions, it'll just use it.
        if self.index_type=='ChromaDB' or self.index_type=='Pinecone':
            self.query_vectorstore=data_processing.initialize_database(self.index_type,
                                                                 self.index_name+'-queries',
                                                                 self.query_model,
                                                                 'Standard',    # Regardless of doc_vectorstore, query_vectorstore is always Standard
                                                                 local_db_path=self.local_db_path,
                                                                 clear=reset_query_db)

        # Initialize retriever
        search_kwargs = _process_retriever_args(self.index_type,
                                                self.search_type,
                                                self.k,
                                                self.fetch_k)
        if self.rag_type=='Standard': 
            if self.index_type=='Pinecone':
                if self.user_doc_namespace:
                    self.retriever=self.doc_vectorstore.as_retriever(search_type=self.search_type,
                                                                     search_kwargs=search_kwargs,
                                                                     namespace=self.user_doc_namespace)
                else:
                    self.retriever=self.doc_vectorstore.as_retriever(search_type=self.search_type,
                                                                     search_kwargs=search_kwargs) 
            if self.index_type=='ChromaDB':
                self.retriever=self.doc_vectorstore.as_retriever(search_type=self.search_type,
                                                                 search_kwargs=search_kwargs)
            elif self.index_type=='RAGatouille':
                self.retriever=self.doc_vectorstore.as_langchain_retriever(k=search_kwargs['k'])  
        elif self.rag_type=='Parent-Child' or self.rag_type=='Summary':
            self.lfs = LocalFileStore(Path(self.local_db_path).resolve() / 'local_file_store' / self.index_name)
            self.retriever = MultiVectorRetriever(
                                vectorstore=self.doc_vectorstore,
                                byte_store=self.lfs,
                                id_key="doc_id",
                                search_type=self.search_type,
                                search_kwargs=search_kwargs)
        else:
            raise NotImplementedError

        # Intialize memory
        self.memory = ConversationBufferMemory(
                        return_messages=True, output_key='answer', input_key='question')

        # Assemble main chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.retriever,
                                                      self.memory)
    def query_docs(self,query): 
        """
        Executes a query and retrieves the relevant documents.

        Args:
            query (str): The query string.

        Returns:
            None
        """       

        # Retrieve memory, invoke chain
        self.memory.load_memory_variables({})

        # Add answer to response, create an array as more prompts come in
        answer_result = self.conversational_qa_chain.invoke({'question': query})
        if not hasattr(self, 'result') or self.result is None:
            self.result = [answer_result]
        else:
            self.result.append(answer_result)

        # Add sources to response, create an array as more prompts come in
        answer_sources = [data.metadata for data in self.result[-1]['references']]
        if not hasattr(self, 'sources') or self.sources is None:
            self.sources = [answer_sources]
        else:
            self.sources.append(answer_sources)

        # Add answer to memory
        if self.llm.__class__.__name__=='ChatOpenAI' or self.llm.__class__.__name__=='ChatAnthropic':
            self.ai_response = self.result[-1]['answer'].content
        else:
            raise NotImplementedError   # To catch any weird stuff I might add later and break the chatbot
        self.memory.save_context({'question': query}, {'answer': self.ai_response})

        # If ChromaDB type, upsert query into query database
        if self.index_type=='ChromaDB' or self.index_type=='Pinecone':
            self.query_vectorstore.add_documents([_question_as_doc(query, self.result[-1])])
        
    def update_model(self,
                     llm: Union[ChatAnthropic, ChatOpenAI]):
        """
        Updates with a new LLM.

        Args:
            llm (ChatAnthropic or ChatOpenAI): The language model for generating responses.

        Returns:
            None
        """
        # TODO add in updated retrieval parameters
        self.llm=llm

        # Update conversational retrieval chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.retriever,
                                                      self.memory)
    def generate_alternative_questions(self,
                                       prompt:str):
        """
        Generates alternative questions based on a prompt.

        Args:
            prompt (str): The prompt for generating alternative questions.

        Returns:
            str: The generated alternative questions.
        """
        if self.ai_response:
            prompt_template=GENERATE_SIMILAR_QUESTIONS_W_CONTEXT
            invoke_dict={'question':prompt,'context':self.ai_response}
        else:
            prompt_template=GENERATE_SIMILAR_QUESTIONS
            invoke_dict={'question':prompt}
        
        chain = (
                prompt_template
                | self.llm
                | StrOutputParser()
            )
        
        alternative_questions=chain.invoke(invoke_dict)
        return alternative_questions

# Internal functions
def _combine_documents(docs, 
                       document_prompt=DEFAULT_DOCUMENT_PROMPT, 
                       document_separator='\n\n'):
    """Combines a list of documents into a single string.

    Args:
        docs (list): A list of documents to be combined.
        document_prompt (str, optional): The prompt to be added before each document.
        document_separator (str, optional): The separator to be added between each document.

    Returns:
        str: The combined string of all the documents.
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def _define_qa_chain(llm,
                     retriever,
                     memory):
    """Defines the conversational QA chain.

    Args:
        llm: The language model component.
        retriever: The document retriever component.
        memory: The memory component.

    Returns:
        The conversational QA chain.

    Raises:
        None.
    """
    # This adds a 'memory' key to the input object
    loaded_memory = RunnablePassthrough.assign(
                        chat_history=RunnableLambda(memory.load_memory_variables) 
                        | itemgetter('history'))  
    
    # Assemble main chain
    standalone_question = {
        'standalone_question': {
            'question': lambda x: x['question'],
            'chat_history': lambda x: get_buffer_string(x['chat_history'])}
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()}
    
    retrieved_documents = {
        'source_documents': itemgetter('standalone_question') 
                            | retriever,
        'question': lambda x: x['standalone_question']}
    
    # Now we construct the inputs for the final prompt
    final_inputs = {
        'context': lambda x: _combine_documents(x['source_documents']),
        'question': itemgetter('question')}
    
    # And finally, we do the part that returns the answers
    answer = {
        'answer': final_inputs 
                    | QA_PROMPT 
                    | llm,
        'references': itemgetter('source_documents')}
    
    conversational_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return conversational_qa_chain
def _process_retriever_args(index_type,
                            search_type='similarity',
                            k=6,
                            fetch_k=50):
    """
    Process the retriever arguments.

    Args:
        index_type (str): The type of index.
        search_type (str, optional): The type of search.
        k (int, optional): The number of documents to retrieve.
        fetch_k (int, optional): The number of documents to fetch.
    Returns:
        dict: The search arguments for the retriever.
    """
    # Set up filter
    if index_type=='Pinecone':
        filter_kwargs={"type": {"$ne": "db_metadata"}}    # Filters out metadata vector
        # if user_doc_namespace:
        #     filter_kwargs = {
        #         "$and": [
        #             {"source_namespace": {"$in": ["default", 'user_upload_736f2c']}},  # Include specific namespaces
        #             {"type": {"$ne": "db_metadata"}}  # Exclude db_metadata vector
        #         ]
        #     }
    else:
        filter_kwargs=None

    # Implement filtering and number of documents to return
    if search_type=='mmr':
        search_kwargs={'k':k,'fetch_k':fetch_k} # See as_retriever docs for parameters
    else:
        search_kwargs={'k':k} # See as_retriever docs for parameters

    retriever_kwargs={'filter':filter_kwargs,
                       **search_kwargs}
    
    return retriever_kwargs
def _question_as_doc(question: str, rag_answer: dict):
    """
    Creates a Document object based on the given question and RAG answer.

    Args:
        question (str): The question to be included in the page content of the Document.
        rag_answer (dict): The RAG answer containing the references and answer content.

    Returns:
        Document: The created Document object.

    """
    sources = [data.metadata for data in rag_answer['references']]

    return Document(
        page_content=question,
        metadata={
            "answer": rag_answer['answer'].content,
            "sources": ",".join(map(data_processing._stable_hash_meta, sources)),
        },
    )