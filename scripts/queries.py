"""
@author: dsmueller3760
Query from pinecone embeddings
"""
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain import PromptTemplate

from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, QA_WSOURCES_PROMPT



class QA_Model:
    def __init__(self, 
                 index_name,
                 embeddings_model,
                 llm,
                 k=6,
                 search_type='similarity',
                 temperature=0,
                 verbose=False,
                 chain_type='stuff',
                 filter=None):
        
        self.index_name:str=index_name
        self.embeddings_model:OpenAIEmbeddings=embeddings_model
        self.llm=llm
        self.k:int=k
        self.search_type:str=search_type
        self.temperature:int=temperature
        self.verbose:bool=verbose
        self.chain_type:str=chain_type
        self.filter:dict=filter

        load_dotenv(find_dotenv(),override=True)

        # Read in from the vector database
        self.vectorstore = Pinecone.from_existing_index(index_name,embeddings_model)

        # Set up question generator and qa with sources
        self.question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=verbose)
        self.doc_chain = load_qa_with_sources_chain(llm, chain_type=chain_type,prompt=QA_WSOURCES_PROMPT,verbose=verbose)

        # Establish chat history
        self.chat_history=ConversationBufferMemory(memory_key='chat_history',
                                            input_key='question',
                                            output_key='answer',
                                            return_messages=True)


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
                    return_generated_question=True)
        
        self.sources=[]

    def query_docs(self,query,filter=None):
        self.result=self.qa({"question": query})

        print('-------------')
        print(query+'\n')
        print(self.result['answer']+'\n\n'+'Sources:'+'\n')

        temp_sources=[]
        for data in self.result['source_documents']:
            temp_sources.append(data.metadata)
            print(data.metadata)

        self.sources.append(temp_sources)
        print('\nGenerated question: '+self.result['generated_question'])
        print('-------------\n')

    def update_model(self,llm,
                    k=6,
                    search_type='similarity',
                    fetch_k=50,
                    verbose=None,
                    filter=None):

        self.llm=llm

        # Set up question generator and qa with sources
        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=verbose)
        self.doc_chain = load_qa_with_sources_chain(self.llm, chain_type=self.chain_type,prompt=QA_WSOURCES_PROMPT,verbose=verbose)

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
            return_generated_question=True)