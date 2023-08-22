"""
@author: dsmueller3760
Query docs and generate responses with different kinds of chains and agents
"""
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def query_docs(index_name,
              embeddings_model,
              query,
              k=4,
              temperature=1,
              verbose=False):
    # Read in from the vector database
    vectorstore = Pinecone.from_existing_index(index_name,embeddings_model)
    
    # Do a similarity search with the query, returning the top k docs.
    docs = vectorstore.similarity_search(query,k=k)

    # Generate a response from the query with the k docs which were found in the database
    llm = OpenAI(temperature=temperature)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=verbose)
    response=chain.run(input_documents=docs, question=query)
    return response, docs