"""QA model and retrieval logic."""

from langchain_core.messages import get_buffer_string

from ..core.cache import Dependencies
from ..services.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT, 
                                DEFAULT_DOCUMENT_PROMPT, GENERATE_SIMILAR_QUESTIONS,
                                GENERATE_SIMILAR_QUESTIONS_W_CONTEXT)

class QAModel:
    """Handles question answering and retrieval."""
    
    def __init__(self,
                 db_service,
                 llm_service=None,
                 k=4,
                 search_type='similarity',
                 namespace=None):
        self.db_service = db_service
        self.llm_service = llm_service
        self.k = k
        self.search_type = search_type
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
        memory_vars = self._memory.load_memory_variables({})
        
        # Get standalone question
        standalone_question = self._get_standalone_question(
            question, 
            memory_vars.get('history', [])
        )
        
        # Retrieve relevant documents
        docs = self._retriever.get_relevant_documents(standalone_question)
        
        # Generate answer using the new chain
        answer_result = self._qa_chain.invoke({'question': standalone_question})
        
        # Update result and sources arrays
        if self.result is None:
            self.result = [answer_result]
        else:
            self.result.append(answer_result)

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
        
        # If ChromaDB type, upsert query into query database
        if self.db_service.db_type in ['ChromaDB', 'Pinecone']:
            self.query_vectorstore.add_documents([self._question_as_doc(question, self.result[-1])])
        
        return {
            'answer': answer_result['answer'],
            'sources': docs,
            'standalone_question': standalone_question
        }
        
    def generate_similar_questions(self, question, n=3):
        """Generate similar questions using the LLM."""
        prompt = GENERATE_SIMILAR_QUESTIONS.format(question=question, n=n)
        response = self.llm_service.get_llm().invoke(prompt)
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        return questions[:n]
        
    def _setup_memory(self):
        """Initialize conversation memory."""
        ConversationBufferMemory = self._deps.get_core_deps()[0]
        self._memory = ConversationBufferMemory(
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
        
    def _setup_retriever(self):
        """Initialize document retriever."""
        search_kwargs = {
            'k': self.k,
            'fetch_k': min(self.k * 4, 100) if self.search_type == 'mmr' else self.k
        }
        
        if self.db_service.db_type == 'pinecone':
            search_kwargs['filter'] = {"type": {"$ne": "db_metadata"}}
            
        vectorstore = self.db_service.initialize_database(
            index_name=self.index_name,
            embedding_service=self.embedding_service,
            namespace=self.namespace
        )
        
        self._retriever = vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs
        )
        
    def _define_qa_chain(self):
        """Defines the conversational QA chain."""
        # Get dependencies
        _, itemgetter, StrOutputParser, RunnableLambda, RunnablePassthrough= self._deps.get_query_deps()
        
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
        format_document, _, _, _, _= self._deps.get_query_deps()

        # Format each document using the format_document function
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings) # Join the formatted strings with the separator
