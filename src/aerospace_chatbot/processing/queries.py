"""QA model and retrieval logic."""

import logging

# Utilities
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import get_buffer_string
from langchain.schema import format_document

# from ..core.cache import Dependencies
from ..services.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT, 
                                DEFAULT_DOCUMENT_PROMPT,
                                GENERATE_SIMILAR_QUESTIONS_W_CONTEXT)
# from ..services.database import DatabaseService
# from ..processing.documents import DocumentProcessor

class QAModel:
    """
    Handles question answering and retrieval.
    """

    def __init__(self,
                 db_service,
                 llm_service,
                 k=8):
        """
        Initialize QA model with necessary services.
        """
        self.db_service = db_service
        self.llm_service = llm_service
        self.k = k
        self.sources = []
        self.scores = []
        self.ai_response = ""
        self.result = []
        self.conversational_qa_chain = None
        self.logger = logging.getLogger(__name__)
        
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
        """
        Executes a query and retrieves the relevant documents.
        """       
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
        answer_scores =  self.result[-1]['scores']
        if not hasattr(self, 'sources') or self.sources is None:
            self.sources = [answer_sources]
            self.scores = [answer_scores]
        else:
            self.sources.append(answer_sources)
            self.scores.append(answer_scores)

        # Add answer to memory
        self.ai_response = self.result[-1]['answer'].content
        self.memory.save_context({'question': query}, {'answer': self.ai_response})


    def generate_alternative_questions(self, prompt):
        """
        Generates alternative questions based on a prompt.
        """
        prompt_template=GENERATE_SIMILAR_QUESTIONS_W_CONTEXT
        invoke_dict={'question':prompt,'context':self.ai_response}
        
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
    
    def _define_qa_chain(self):
        """
        Defines the conversational QA chain.
        """
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
        
        retrieval_results = RunnablePassthrough.assign(
            retrieval=lambda x: self.db_service.retriever.invoke(x['standalone_question'])
        )

        retrieved_documents = RunnablePassthrough.assign(
            source_documents=lambda x: x['retrieval'][0],  # Get docs from first element of tuple
            scores=lambda x: x['retrieval'][1]             # Get scores from second element of tuple
        )
        
        final_inputs = {
            'context': lambda x: self._combine_documents(x['source_documents']),
            'question': itemgetter('standalone_question')}
        
        answer = {
            'answer': final_inputs 
                        | QA_PROMPT 
                        | self.llm_service.get_llm(),
            'references': itemgetter('source_documents'),
            'scores': itemgetter('scores')}
        
        return loaded_memory | standalone_question | retrieval_results | retrieved_documents | answer
    
    def _combine_documents(self, docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator='\n\n'):
        """
        Combines a list of documents into a single string using the format_document function.
        """        
        # Format each document using the cached format_document function
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        
        # Join the formatted strings with the separator
        return document_separator.join(doc_strings)

    def _get_standalone_question(self, question, chat_history):
        """
        Generate standalone question from conversation context.
        """
        if not chat_history:
            return question
            
        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=get_buffer_string(chat_history),
            question=question
        )
        
        response = self.llm_service.get_llm().invoke(prompt)
        return response.content.strip()