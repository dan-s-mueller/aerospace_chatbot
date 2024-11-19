"""QA model and retrieval logic."""

import logging

from ..core.cache import Dependencies
from ..services.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT, 
                                DEFAULT_DOCUMENT_PROMPT, GENERATE_SIMILAR_QUESTIONS,
                                GENERATE_SIMILAR_QUESTIONS_W_CONTEXT)
from ..services.database import DatabaseService
from ..processing.documents import DocumentProcessor

class QAModel:
    """Handles question answering and retrieval."""
    
    def __init__(self,
                 db_service,
                 llm_service,
                 k=8):
        """Initialize QA model with necessary services."""
        self.db_service = db_service
        self.llm_service = llm_service
        self.k = k
        self.sources = []
        self.ai_response = ""
        self.result = []
        self.conversational_qa_chain = None
        self.logger = logging.getLogger(__name__)

        # Get chain utilities
        _, _, _, _, ConversationBufferMemory, _, _, _ = Dependencies.LLM.get_chain_utils()

        # Create a separate database service for query storage
        if self.db_service.db_type in ['ChromaDB', 'Pinecone']:
            self.query_db_service = DatabaseService(
                db_type=self.db_service.db_type,
                index_name=self.db_service.index_name,
                rag_type="Standard",
                embedding_service=self.db_service.embedding_service,
                doc_type='question'
            )
            self.query_db_service.initialize_database(clear=False)   # TODO decide if this should clear every time
        
        # Get retrievers from database services
        self.db_service.get_retriever(k=k)

        # Initialize memory
        self.memory = ConversationBufferMemory(
            return_messages=True, 
            output_key='answer', 
            input_key='question'
        )
        self.conversational_qa_chain = self._define_qa_chain()

        if self.conversational_qa_chain is None:
            raise ValueError("QA chain not initialized")
        
    def query(self,query): 
        """Executes a query and retrieves the relevant documents."""       
        # Retrieve memory, invoke chain
        self.memory.load_memory_variables({})

        # Add answer to response, create an array as more prompts come in
        self.logger.info(f'Invoking QA chain with query: {query}')
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

        # If compatible type, upsert query into query database
        if self.db_service.db_type in ['ChromaDB', 'Pinecone']:
            self.logger.info(f'Upserting question into query database {self.query_db_service.index_name}')
            self.query_db_service.index_data(data=[self._question_as_doc(query, self.result[-1])])
    def generate_alternative_questions(self, prompt):
        """Generates alternative questions based on a prompt."""
        _, StrOutputParser, _, _, _, _, _, _ = Dependencies.LLM.get_chain_utils()
        if self.ai_response:
            prompt_template=GENERATE_SIMILAR_QUESTIONS_W_CONTEXT
            invoke_dict={'question':prompt,'context':self.ai_response}
        else:
            prompt_template=GENERATE_SIMILAR_QUESTIONS
            invoke_dict={'question':prompt}
        
        chain = (
                prompt_template
                | self.llm_service.get_llm()
                | StrOutputParser()
            )
        alternative_questions = chain.invoke(invoke_dict)
        self.logger.info(f'Generated alternative questions: {alternative_questions}')
        # Split the string into a list of questions, removing empty strings and stripping whitespace
        alternative_questions = [question.strip() for question in alternative_questions.split('\n') if question.strip()]
        self.logger.info(f'Alternative questions split up: {alternative_questions}')
        return alternative_questions
    def _setup_memory(self):
        """Initialize conversation memory."""
        _, _, _, _, ConversationBufferMemory, _, _, _ = Dependencies.LLM.get_chain_utils()
        self.memory = ConversationBufferMemory(
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
    def _define_qa_chain(self):
        """Defines the conversational QA chain."""
        itemgetter, StrOutputParser, RunnableLambda, RunnablePassthrough, _, get_buffer_string, _, _ = Dependencies.LLM.get_chain_utils()
        
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
                                | self.db_service.retriever,
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
        _, _, _, _, _, _, _, format_document = Dependencies.LLM.get_chain_utils()
        
        # Format each document using the cached format_document function
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        
        # Join the formatted strings with the separator
        return document_separator.join(doc_strings)
    @staticmethod
    def _question_as_doc(question, rag_answer):
        """Creates a Document object based on the given question and RAG answer."""
        _, _, _, _, _, _, Document, _ = Dependencies.LLM.get_chain_utils()

        # TODO this feels really fragile, but it's the best I can think of for now.
        for i, doc in enumerate(rag_answer['references']):
            for key, value in doc.metadata.items():
                if isinstance(value, float) and not isinstance(value, (bool, int, str)):
                    doc.metadata[key] = int(value)
                    rag_answer['references'][i] = doc

        sources = [DocumentProcessor.stable_hash_meta(doc.metadata) for doc in rag_answer['references']]
        return Document(
            page_content=question,
            metadata={
                "answer": rag_answer['answer'].content,
                "sources": ','.join(sources),  # Now sources is a list of IDs, no need to join
            },
        )
    def _get_standalone_question(self, question, chat_history):
        """Generate standalone question from conversation context."""
        _, _, _, _, _, get_buffer_string, _, _ = Dependencies.LLM.get_chain_utils()

        if not chat_history:
            return question
            
        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=get_buffer_string(chat_history),
            question=question
        )
        
        response = self.llm_service.get_llm().invoke(prompt)
        return response.content.strip()