import os
import time
import logging

from dotenv import load_dotenv, find_dotenv

import pinecone

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

# import langchain_openai.OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

# import langchain_openai.OpenAI
from langchain_community.llms import OpenAI
from langchain_community.llms import HuggingFaceHub

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain

from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, QA_WSOURCES_PROMPT

# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

class QA_Model:
    def __init__(self, 
                 index_type,
                 index_name,
                 embeddings_model,
                 llm,
                 k=6,
                 search_type='similarity',
                 temperature=0,
                 verbose=False,
                 chain_type='stuff',
                 filter_arg=False):
        
        self.index_type:str=index_type
        self.index_name:str=index_name
        self.embeddings_model:OpenAIEmbeddings=embeddings_model
        self.llm=llm
        self.k:int=k
        self.search_type:str=search_type
        self.temperature:int=temperature
        self.verbose:bool=verbose
        self.chain_type:str=chain_type
        self.filter_arg:bool=filter_arg

        load_dotenv(find_dotenv(),override=True)

        # Read in from the vector database
        if index_type=='Pinecone':
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )
            logging.info('Pinecone index name: '+str(index_name))
            logging.info('Embedding model: '+str(embeddings_model))
            self.vectorstore = Pinecone.from_existing_index(index_name,embeddings_model)
            logging.info('Vectorstore: '+str(self.vectorstore))
        elif index_type=='ChromaDB':
            logging.info('Chroma index name: '+str(index_name))
            logging.info('Embedding model: '+str(embeddings_model))
            self.vectorstore = Chroma(persist_directory=f'../db/{index_name}',
                                      embedding_function=embeddings_model)
            logging.info('Vectorstore: '+str(self.vectorstore))
        elif index_type=='RAGatouille':
            raise NotImplementedError
        
        # Set up question generator and qa with sources
        self.question_generator = LLMChain(llm=llm, 
                                           prompt=CONDENSE_QUESTION_PROMPT,
                                           verbose=verbose)
        logging.info('Question generator: '+str(self.question_generator))
        self.doc_chain = load_qa_with_sources_chain(llm, chain_type=chain_type,prompt=QA_WSOURCES_PROMPT,verbose=verbose)
        logging.info('Doc chain: '+str(self.doc_chain))

        # Establish chat history
        self.chat_history=ConversationBufferMemory(memory_key='chat_history',
                                            input_key='question',
                                            output_key='answer',
                                            return_messages=True)
        logging.info('Chat history: '+str(self.chat_history))

        # Implement filter
        if filter_arg:
            logging.info('Filtering sources')
            filter_list = list(set(item["source"] for item in self.sources[-1]))
            filter_items=[]
            for item in filter_list:
                filter_item={"source": item}
                filter_items.append(filter_item)
            filter={"$or":filter_items}
            logging.info('Filter: '+str(filter))
        else:
            filter=None

        if search_type=='mmr':
            search_kwargs={'k':k,'fetch_k':50,'filter':filter} # See as_retriever docs for parameters
        else:
            search_kwargs={'k':k,'filter':filter} # See as_retriever docs for parameters

        self.qa = ConversationalRetrievalChain(
                    retriever=self.vectorstore.as_retriever(search_type=search_type,
                                                            search_kwargs=search_kwargs),  
                    combine_docs_chain=self.doc_chain, 
                    question_generator=self.question_generator,
                    memory=self.chat_history,
                    verbose=verbose,
                    return_source_documents=True,
                    return_generated_question=True,
                    )
        logging.info('ConversationalRetrieverChain: '+str(self.qa))   
        
        self.sources=[]

    def query_docs(self,query,tags=None):
        self.result=self.qa({'question': query},tags=tags)

        # print('-------------')
        # print(query+'\n')
        # print(self.result['answer']+'\n\n'+'Sources:'+'\n')

        temp_sources=[]
        for data in self.result['source_documents']:
            temp_sources.append(data.metadata)
            # print(data.metadata)

        self.sources.append(temp_sources)
        # print('\nGenerated question: '+self.result['generated_question'])
        # print('-------------\n')

    def update_model(self,llm,
                    k=6,
                    search_type='similarity',
                    fetch_k=50,
                    verbose=None,
                    filter_arg=False):

        self.llm=llm

        # Set up question generator and qa with sources
        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=verbose)
        self.doc_chain = load_qa_with_sources_chain(self.llm, chain_type=self.chain_type,prompt=QA_WSOURCES_PROMPT,verbose=verbose)

        # Implement filter
        if filter_arg:
            print(self.sources)
            filter_list = list(set(item["source"] for item in self.sources[-1]))
            filter_items=[]
            for item in filter_list:
                filter_item={"source": item}
                filter_items.append(filter_item)
            filter={"$or":filter_items}
        else:
            filter=None

        if search_type=='mmr':
            search_kwargs={'k':k,'fetch_k':fetch_k,'filter':filter} # See as_retriever docs for parameters
        else:
            search_kwargs={'k':k,'filter':filter} # See as_retriever docs for parameters

        self.qa = ConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever(search_type=search_type,
                                                            search_kwargs=search_kwargs),
            combine_docs_chain=self.doc_chain, 
            question_generator=self.question_generator,
            memory=self.chat_history,
            verbose=verbose,
            return_source_documents=True,
            return_generated_question=True,
            )