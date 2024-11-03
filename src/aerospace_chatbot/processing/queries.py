"""QA model and retrieval logic."""

from pathlib import Path
from langchain_core.messages import get_buffer_string

from ..core.cache import Dependencies
from ..services.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT, 
                                DEFAULT_DOCUMENT_PROMPT, GENERATE_SIMILAR_QUESTIONS,
                                GENERATE_SIMILAR_QUESTIONS_W_CONTEXT)

class QAModel:
    """Handles question answering and retrieval."""
    
    def __init__(self,
                 db_service,
                 llm_service,
                 k=4,
                 namespace=None):
        self.db_service = db_service
        self.llm_service = llm_service
        self.k = k
        self.namespace = namespace
        self._deps = Dependencies()
        self._memory = None
        self._retriever = None
        self._qa_chain = None
        self.result = None
        self.sources = None
        self.ai_response = None
        self.query_vectorstore = None
    def query(self, question):
        """Process a query and return answer with sources."""
        # Initialize components if needed
        if self._memory is None:
            self._setup_memory()
        if self._retriever is None:
            self._setup_retriever()
        if self._qa_chain is None:
            self._define_qa_chain()
            
        # Get conversation history
        self._memory.load_memory_variables({})
        
        # Add answer to response, create an array as more prompts come in
        answer_result = self._qa_chain.invoke({'question': question})
        if self.result is None:
            self.result = [answer_result]
        else:
            self.result.append(answer_result)

        # Add sources to response, create an array as more prompts come in
        answer_sources = [data.metadata for data in self.result[-1]['references']]
        if self.sources is None:
            self.sources = [answer_sources]
        else:
            self.sources.append(answer_sources)

        # Process answer based on LLM type
        if self.llm_service.get_llm().__class__.__name__ in ['ChatOpenAI', 'ChatAnthropic']:
            self.ai_response = self.result[-1]['answer'].content
        else:
            raise NotImplementedError
        
        # Update memory
        self._memory.save_context(
            {'question': question},
            {'answer': self.ai_response}
        )
        
        # Upsert query into query database if compatible
        if self.db_service.db_type in ['ChromaDB', 'Pinecone']:
            self.query_vectorstore.add_documents([self._question_as_doc(question, self.result[-1])])
        
        return {
            'answer': answer_result['answer'],
            'sources': answer_sources
        }
    def generate_similar_questions(self, question, n=3):
        """Generate similar questions using the LLM."""
        prompt = GENERATE_SIMILAR_QUESTIONS.format(question=question, n=n)
        response = self.llm_service.get_llm().invoke(prompt)
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        return questions[:n]
    def _setup_memory(self):
        """Initialize conversation memory."""
        _, _, _, _, ConversationBufferMemory= self._deps.get_query_deps()
        self._memory = ConversationBufferMemory(
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
    def _setup_retriever(self):
        """Initialize document retriever."""
        search_kwargs = self._process_retriever_args(
            self.db_service.db_type,  
            self.k
        )

        # Get the vectorstore directly from db_service
        vectorstore = self.db_service.vectorstore
        if not vectorstore:
            raise ValueError("Database not initialized. Please ensure database is initialized before setting up retriever.")

        if self.db_service.rag_type == 'Standard':
            self._setup_standard_retriever(vectorstore, search_kwargs)
        elif self.db_service.rag_type in ['Parent-Child', 'Summary']:
            self._setup_multivector_retriever(vectorstore, search_kwargs)
        else:
            raise NotImplementedError(f"RAG type {self.db_service.rag_type} not supported")
    def _setup_standard_retriever(self, vectorstore, search_kwargs):
        """Set up standard retriever based on index type."""
        if self.db_service.db_type in ['Pinecone', 'ChromaDB']:
            self._retriever = vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs=search_kwargs
            )
        elif self.db_service.db_type == 'RAGatouille':
            self._retriever = vectorstore.as_langchain_retriever(
                k=search_kwargs['k']
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_service.db_type}")
    def _setup_multivector_retriever(self, vectorstore, search_kwargs):
        """Set up multi-vector retriever for Parent-Child or Summary RAG types."""
        LocalFileStore = self._deps.get_core_deps()[2]  # Get LocalFileStore from dependencies
        MultiVectorRetriever = self._deps.get_core_deps()[1]  # Get MultiVectorRetriever from dependencies
        
        self.lfs = LocalFileStore(
            Path(self.local_db_path).resolve() / 'local_file_store' / self.db_service.index_name
        )
        self._retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=self.lfs,
            id_key="doc_id",
            search_kwargs=search_kwargs
        )
    @staticmethod
    def _process_retriever_args(db_type, k=4):
        """Process the retriever arguments."""
        # Set up filter
        if db_type == 'Pinecone':
            filter_kwargs = {"type": {"$ne": "db_metadata"}}
        else:
            filter_kwargs = None

        # Implement filtering and number of documents to return
        search_kwargs = {'k': k}
        
        if filter_kwargs:
            search_kwargs['filter'] = filter_kwargs
            
        return search_kwargs
    def _define_qa_chain(self):
        """Defines the conversational QA chain."""
        # Get dependencies
        itemgetter, StrOutputParser, RunnableLambda, RunnablePassthrough, _= self._deps.get_query_deps()
        
        # This adds a 'memory' key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self._memory.load_memory_variables) 
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
                                | self._retriever,
            'question': lambda x: x['standalone_question']}
        
        final_inputs = {
            'context': lambda x: self._combine_documents(x['source_documents']),
            'question': itemgetter('question')}
        
        answer = {
            'answer': final_inputs 
                        | QA_PROMPT 
                        | self.llm_service.get_llm(),
            'references': itemgetter('source_documents')}
        
        self._qa_chain = loaded_memory | standalone_question | retrieved_documents | answer
    def _get_standalone_question(self, question, chat_history):
        """Generate standalone question from conversation context."""
        if not chat_history:
            return question
            
        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=get_buffer_string(chat_history),
            question=question
        )
        
        response = self.llm_service.get_llm().invoke(prompt)
        return response.content.strip()
    def _combine_documents(self, docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator='\n\n'):
        """Combines a list of documents into a single string using the format_document function."""
        from langchain.schema import format_document

        # Format each document using the format_document function
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings) # Join the formatted strings with the separator
