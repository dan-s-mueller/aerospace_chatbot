"""QA model and retrieval logic."""

from pathlib import Path
from typing import Dict
from datetime import datetime

from ..core.cache import Dependencies
from ..services.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT, 
                                DEFAULT_DOCUMENT_PROMPT, GENERATE_SIMILAR_QUESTIONS,
                                GENERATE_SIMILAR_QUESTIONS_W_CONTEXT)
from ..services.database import DatabaseService
from ..processing.documents import DocumentProcessor
from langchain_core.documents import Document

class QAModel:
    """Handles question answering and retrieval."""
    
    def __init__(self,
                 db_service,
                 llm_service,
                 k=4,
                 namespace=None):
        """Initialize QA model with necessary services."""
        print("Initializing QAModel...")  # Debug print
        self.db_service = db_service
        self.llm_service = llm_service
        self.k = k
        self.namespace = namespace
        self._deps = Dependencies()
        self.sources = []
        self.ai_response = ""
        self.result = []
        self.conversational_qa_chain = None  # Explicitly initialize as None

        _, _, _, _, ConversationBufferMemory, _ = self._deps.get_query_deps()

        # Create a separate database service for query storage
        self.query_db_service = DatabaseService(
            db_type=self.db_service.db_type,
            local_db_path=self.db_service.local_db_path
        )
        
        # Initialize query vectorstore with the same embedding service as main db
        self.query_db_service.initialize_database(
            index_name=self.db_service.index_name + '-queries',  # separate index for queries, append -queries
            embedding_service=self.db_service.embedding_service,
            rag_type="Standard",
            namespace=self.namespace
        )
        
        # Get retrievers from database services
        print("Getting retrievers...")  # Debug print
        self.retriever = self.db_service.get_retriever(k=k)
        print(f"Retriever initialized: {self.retriever}")  # Debug print

        # Initialize memory
        print("Initializing memory...")  # Debug print
        self.memory = ConversationBufferMemory(
            return_messages=True, 
            output_key='answer', 
            input_key='question'
        )
        self.conversational_qa_chain = self._define_qa_chain()
        print(f"Chain assigned to self._qa_chain: {self.conversational_qa_chain}")  # Debug print

        if self.conversational_qa_chain is None:
            raise ValueError("QA chain not initialized")
        
    def query(self,query): 
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
        if self.llm_service.get_llm().__class__.__name__=='ChatOpenAI' or self.llm_service.get_llm().__class__.__name__=='ChatAnthropic':
            self.ai_response = self.result[-1]['answer'].content
        else:
            raise NotImplementedError   # To catch any weird stuff I might add later and break the chatbot
        self.memory.save_context({'question': query}, {'answer': self.ai_response})

        # If ChromaDB type, upsert query into query database
        if self.db_service.db_type in ['ChromaDB', 'Pinecone']:
            self.query_vectorstore.add_documents([self._question_as_doc(query, self.result[-1])])
    def _setup_memory(self):
        """Initialize conversation memory."""
        _, _, _, _, ConversationBufferMemory, _= self._deps.get_query_deps()
        self.memory = ConversationBufferMemory(
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
    def _define_qa_chain(self):
        """Defines the conversational QA chain."""
        print("Inside _define_qa_chain...")  # Debug print
        
        # Get dependencies
        itemgetter, StrOutputParser, RunnableLambda, RunnablePassthrough, _, get_buffer_string = self._deps.get_query_deps()
        print("Dependencies retrieved")  # Debug print
        
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
            | self.llm_service.get_llm()
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
                        | self.llm_service.get_llm(),
            'references': itemgetter('source_documents')}
        
        return loaded_memory | standalone_question | retrieved_documents | answer
    def _combine_documents(self, docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator='\n\n'):
        """Combines a list of documents into a single string using the format_document function."""
        # Format each document using the cached format_document function
        from langchain.schema import format_document
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        
        # Join the formatted strings with the separator
        return document_separator.join(doc_strings)
    def _question_as_doc(question: str, rag_answer: dict):
        """Creates a Document object based on the given question and RAG answer."""
        sources = [data.metadata for data in rag_answer['references']]
        return Document(
            page_content=question,
            metadata={
                "answer": rag_answer['answer'].content,
                "sources": ",".join(map(DocumentProcessor._stable_hash_meta, sources)),
            },
        )
    def generate_similar_questions(self, question, n=3):
        """Generate similar questions using the LLM."""
        prompt = GENERATE_SIMILAR_QUESTIONS.format(question=question, n=n)
        response = self.llm_service.get_llm().invoke(prompt)
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        return questions[:n]
    def _get_standalone_question(self, question, chat_history):
        """Generate standalone question from conversation context."""
        _, _, _, _, _, get_buffer_string = self._deps.get_query_deps()

        if not chat_history:
            return question
            
        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=get_buffer_string(chat_history),
            question=question
        )
        
        response = self.llm_service.get_llm().invoke(prompt)
        return response.content.strip()