'''
@author: dsmueller3760
Query docs and generate responses with different kinds of chains and agents
'''
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain import PromptTemplate
from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, QA_WSOURCES_PROMPT


def qa_model(index_name,
              embeddings_model,
              llm,
              k=6,
              search_type='similarity',
              temperature=0,
              verbose=False,
              chain_type='stuff'):
    
    load_dotenv(find_dotenv(),override=True)

    # Read in from the vector database
    vectorstore = Pinecone.from_existing_index(index_name,embeddings_model)
    
    # Do a similarity search with the query, returning the top k docs.
    # docs = vectorstore.similarity_search(query,k=k)

    llm = OpenAI(temperature=0)

    # Generate a response from the query with the k docs which were found in the database
    # chain = load_qa_chain(llm, chain_type='stuff', verbose=verbose)
    # response=chain.run(input_documents=docs, question=query)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=verbose)
    # doc_chain = load_qa_chain(llm, chain_type=chain_type, prompt=QA_PROMPT,verbose=verbose)
    doc_chain = load_qa_with_sources_chain(llm, chain_type='stuff',prompt=QA_WSOURCES_PROMPT,verbose=verbose)

    chat_history=ConversationBufferMemory(memory_key='chat_history',
                                          input_key='question',
                                          output_key='answer',
                                          return_messages=True)


    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_type=search_type,
                                            search_kwargs={'k':k,'fetch_k': 50}),  # See as_retriever docs for parameters
                                            combine_docs_chain=doc_chain, 
                                            question_generator=question_generator,
                                            memory=chat_history,
                                            return_source_documents=True,
                                            verbose=verbose,
                                            return_generated_question=True)

    # query = 'What can you tell me about latch mechanism design failures which have occurred'
    # result = qa({"question": query})

    # query_followup='Provide details on the inadequate engineering controls on critical features'
    # result = qa({"question": query_followup})

    return qa

def query_docs(qa,query,filter=None):
    result=qa({"question": query})

    print(query+'\n')
    print(result['answer']+'\n\n'+'Sources:'+'\n')

    for data in result['source_documents']:
        print(data.metadata)

    print('\nGenerated question: '+result['generated_question'])
    print('-------------\n')