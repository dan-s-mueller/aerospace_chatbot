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
                 namespace:str=None,
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
            namespace (str, optional): The namespace for user documents.
            reset_query_db (bool, optional): Whether to reset the query database.
        """
        # Directly assign instance attributes
        self.index_type = index_type
        self.index_name = index_name
        self.query_model = query_model
        self.llm = llm
        self.rag_type = rag_type
        self.k = k
        self.search_type = search_type
        self.fetch_k = fetch_k
        self.temperature = temperature
        self.local_db_path = local_db_path
        self.namespace = namespace
        
        # Initialize state attributes
        self._initialize_state()
        
        # Set up vector stores and retriever
        self._setup_vectorstores(reset_query_db)
        self._setup_retriever()
        
        # Initialize memory and chain
        self._setup_memory_and_chain()

    def query_docs(self,query): 
        """Executes a query and retrieves the relevant documents."""       
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
        if self.index_type in ['ChromaDB', 'Pinecone']:
            self.query_vectorstore.add_documents([self._question_as_doc(query, self.result[-1])])
    def update_model(self,
                     llm: Union[ChatAnthropic, ChatOpenAI]):
        """Updates with a new LLM."""
        # TODO add in updated retrieval parameters
        self.llm=llm

        # Update conversational retrieval chain
        self.conversational_qa_chain=self._define_qa_chain()
    def generate_alternative_questions(self,
                                       prompt:str):
        """Generates alternative questions based on a prompt."""
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
    def _initialize_state(self):
        """Initialize all state-related attributes to None."""
        self.memory = None
        self.result = None
        self.sources = None
        self.ai_response = None
        self.conversational_qa_chain = None
        self.doc_vectorstore = None
        self.query_vectorstore = None
        self.retriever = None

    def _setup_vectorstores(self, reset_query_db):
        """Initialize document and query vectorstores."""
        # Document vectorstore
        self.doc_vectorstore = data_processing.initialize_database(
            self.index_type, self.index_name, self.query_model, 
            self.rag_type, local_db_path=self.local_db_path,
            init_ragatouille=False, namespace=self.namespace
        )
        
        # Query vectorstore (if applicable)
        if self.index_type in ['ChromaDB', 'Pinecone']:
            self.query_vectorstore = data_processing.initialize_database(
                self.index_type, f"{self.index_name}-queries",
                self.query_model, 'Standard',
                local_db_path=self.local_db_path,
                clear=reset_query_db
            )

    def _setup_retriever(self):
        """Initialize the appropriate retriever based on RAG type."""
        search_kwargs = self._process_retriever_args(
            self.index_type, self.search_type, self.k, self.fetch_k
        )

        if self.rag_type == 'Standard':
            self._setup_standard_retriever(search_kwargs)
        elif self.rag_type in ['Parent-Child', 'Summary']:
            self._setup_multivector_retriever(search_kwargs)
        else:
            raise NotImplementedError(f"RAG type {self.rag_type} not supported")

    def _setup_standard_retriever(self, search_kwargs):
        """Set up standard retriever based on index type."""
        if self.index_type in ['Pinecone', 'ChromaDB']:
            self.retriever = self.doc_vectorstore.as_retriever(
                search_type=self.search_type,
                search_kwargs=search_kwargs
            )
        elif self.index_type == 'RAGatouille':
            self.retriever = self.doc_vectorstore.as_langchain_retriever(
                k=search_kwargs['k']
            )

    def _setup_multivector_retriever(self, search_kwargs):
        """Set up multi-vector retriever for Parent-Child or Summary RAG types."""
        self.lfs = LocalFileStore(
            Path(self.local_db_path).resolve() / 'local_file_store' / self.index_name
        )
        self.retriever = MultiVectorRetriever(
            vectorstore=self.doc_vectorstore,
            byte_store=self.lfs,
            id_key="doc_id",
            search_type=self.search_type,
            search_kwargs=search_kwargs
        )

    def _setup_memory_and_chain(self):
        """Initialize memory and conversational QA chain."""
        self.memory = ConversationBufferMemory(
            return_messages=True, 
            output_key='answer', 
            input_key='question'
        )
        self.conversational_qa_chain = self._define_qa_chain()
    def _define_qa_chain(self):
        """Defines the conversational QA chain."""
        # This adds a 'memory' key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) 
            | itemgetter('history'))  
        
        # Assemble main chain
        standalone_question = {
            'standalone_question': {
                'question': lambda x: x['question'],
                'chat_history': lambda x: get_buffer_string(x['chat_history'])}
            | CONDENSE_QUESTION_PROMPT
            | self.llm
            | StrOutputParser()}
        
        retrieved_documents = {
            'source_documents': itemgetter('standalone_question') 
                                | self.retriever,
            'question': lambda x: x['standalone_question']}
        
        final_inputs = {
            'context': lambda x: self._combine_documents(x['source_documents']),
            'question': itemgetter('question')}
        
        answer = {
            'answer': final_inputs 
                        | QA_PROMPT 
                        | self.llm,
            'references': itemgetter('source_documents')}
        
        return loaded_memory | standalone_question | retrieved_documents | answer
    @staticmethod
    def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator='\n\n'):
        """Combines a list of documents into a single string."""
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    @staticmethod
    def _process_retriever_args(index_type, search_type='similarity', k=6, fetch_k=50):
        """Process the retriever arguments."""
        # Set up filter
        if index_type == 'Pinecone':
            filter_kwargs = {"type": {"$ne": "db_metadata"}}
        else:
            filter_kwargs = None

        # Implement filtering and number of documents to return
        search_kwargs = {'k': k, 'fetch_k': fetch_k} if search_type == 'mmr' else {'k': k}
        
        return {'filter': filter_kwargs, **search_kwargs}

    @staticmethod
    def _question_as_doc(question: str, rag_answer: dict):
        """Creates a Document object based on the given question and RAG answer."""
        sources = [data.metadata for data in rag_answer['references']]
        return Document(
            page_content=question,
            metadata={
                "answer": rag_answer['answer'].content,
                "sources": ",".join(map(data_processing._stable_hash_meta, sources)),
            },
        )