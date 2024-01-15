import os
import time
import logging

from dotenv import load_dotenv, find_dotenv

import pinecone

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain

# For LCEL upgrade
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema.runnable import RunnableMap, RunnableParallel
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, QA_WSOURCES_PROMPT, DEFAULT_DOCUMENT_PROMPT

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

# Class and functions
class QA_Model:
    def __init__(self, 
                 index_type,
                 index_name,
                 embeddings_model,
                 llm,
                 k=6,
                 search_type='similarity',
                 fetch_k=50,
                 temperature=0,
                 verbose=False,
                 chain_type='stuff',
                 filter_arg=False):
        
        self.index_type=index_type
        self.index_name=index_name
        self.embeddings_model=embeddings_model
        self.llm=llm
        self.k=k
        self.search_type=search_type
        self.fetch_k=fetch_k
        self.temperature=temperature
        self.verbose=verbose
        self.chain_type=chain_type
        self.filter_arg=filter_arg
        self.sources=[]

        load_dotenv(find_dotenv(),override=True)

        # Read in from the vector database
        if index_type=='Pinecone':
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )
            logging.info('Chat pinecone index name: '+str(index_name))
            logging.info('Chat embedding model: '+str(embeddings_model))
            self.vectorstore = Pinecone.from_existing_index(index_name,embeddings_model)
            logging.info('Chat vectorstore: '+str(self.vectorstore))
        elif index_type=='ChromaDB':
            logging.info('Chat chroma index name: '+str(index_name))
            logging.info('Chat embedding model: '+str(embeddings_model))
            self.vectorstore = Chroma(persist_directory=f'../db/{index_name}',
                                      embedding_function=embeddings_model)
            logging.info('Chat vectorstore: '+str(self.vectorstore))
        elif index_type=='RAGatouille':
            raise NotImplementedError

        # Define retriever search parameters
        search_kwargs = _process_retriever_args(self.filter_arg,
                                                self.sources,
                                                self.search_type,
                                                self.k,
                                                self.fetch_k)

        # Intialize memory
        self.memory = ConversationBufferMemory(
                        return_messages=True, output_key="answer", input_key="question")
        logging.info('Memory: '+str(self.memory))

        # Assemble main chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.vectorstore,
                                                      self.memory,
                                                      self.search_type,
                                                      search_kwargs)

        # Usage
        # inputs = {"question": "where did harrison work?"}
        # result = final_chain.invoke(inputs)
        # result
        # self.memory.save_context(inputs, {"answer": result["answer"].content})

    def query_docs(self,query,tags=None):
        # TODO: figure out where to put tags

        self.memory.load_memory_variables({})
        logging.info('Memory content before qa result: '+str(self.memory.content))

        logging.info('Query: '+str(query))
        self.result = self.conversational_qa_chain.invoke({'question': query})
        logging.info('QA result: '+str(self.result))

        self.memory.save_context({'question': query}, {"answer": self.result["answer"].content})
        logging.info('Memory content after qa result: '+str(self.memory.content))

        temp_sources=[]
        for data in self.result['source_documents']:
            temp_sources.append(data.page_content)
            logging.info('Source: '+str(data.page_content))

        self.sources.append(temp_sources)

    def update_model(self,
                     llm,
                     k=6,
                     search_type='similarity',
                     fetch_k=50,
                     verbose=None,
                     filter_arg=False):

        self.llm=llm
        self.k=k
        self.search_type=search_type
        self.fetch_k=fetch_k
        self.verbose=verbose
        self.filter_arg=filter_arg

        # Set up question generator and qa with sources
        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=self.verbose)
        self.doc_chain = load_qa_with_sources_chain(self.llm, chain_type=self.chain_type,prompt=QA_WSOURCES_PROMPT,verbose=self.verbose)

        # Define retriever search parameters
        search_kwargs = _process_retriever_args(self.filter_arg,
                                                self.sources,
                                                self.search_type,
                                                self.k,
                                                self.fetch_k)

        # Update conversational retrieval chain
        self.conversational_qa_chain=_define_qa_chain(self.llm,
                                                      self.vectorstore,
                                                      self.memory,
                                                      self.search_type,
                                                      search_kwargs)

# Internal functions
def _combine_documents(docs, 
                        document_prompt=DEFAULT_DOCUMENT_PROMPT, 
                        document_separator="\n\n"):
    # TODO: this would be where stuff, map reduce, etc. would go
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
def _define_qa_chain(llm,
                     vectorstore,
                     memory,
                     search_type,
                     search_kwargs):
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
                        chat_history=RunnableLambda(memory.load_memory_variables) 
                        | itemgetter("history"))  
    logging.info('Loaded memory: '+str(loaded_memory))
    
    # Assemble main chain
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"])}
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()}
    logging.info('Condense inputs as a standalong question: '+str(standalone_question))
    retrieved_documents = {
        "source_documents": itemgetter("standalone_question") 
                            | vectorstore.as_retriever(search_type=search_type,
                                                       search_kwargs=search_kwargs),
        "question": lambda x: x["standalone_question"]}
    logging.info('Retrieved documents: '+str(retrieved_documents))
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question")}
    logging.info('Combined documents: '+str(final_inputs))
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs 
                    | QA_PROMPT 
                    | llm,
        "docs": itemgetter("docs")}
    conversational_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer
    logging.info('Conversational QA chain: '+str(conversational_qa_chain))
    return conversational_qa_chain
def _process_retriever_args(filter_arg,
                            sources,
                            search_type,
                            k,
                            fetch_k):
    # Implement filter
    if filter_arg:
        filter_list = list(set(item["source"] for item in sources[-1]))
        filter_items=[]
        for item in filter_list:
            filter_item={"source": item}
            filter_items.append(filter_item)
        filter={"$or":filter_items}
    else:
        filter=None

    # Impement filtering and number of documents to return
    if search_type=='mmr':
        search_kwargs={'k':k,'fetch_k':fetch_k,'filter':filter} # See as_retriever docs for parameters
    else:
        search_kwargs={'k':k,'filter':filter} # See as_retriever docs for parameters
    
    return search_kwargs