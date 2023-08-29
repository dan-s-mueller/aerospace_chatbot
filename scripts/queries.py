"""
@author: dsmueller3760
Query docs and generate responses with different kinds of chains and agents
"""
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate

def query_docs(index_name,
              embeddings_model,
              query,
              k=6,
              temperature=.5,
              verbose=False):
    # Read in from the vector database
    vectorstore = Pinecone.from_existing_index(index_name,embeddings_model)
    
    # Do a similarity search with the query, returning the top k docs.
    docs = vectorstore.similarity_search(query,k=k)

    # Generate a response from the query with the k docs which were found in the database
    llm = OpenAI(temperature=temperature)
    # chain = load_qa_chain(llm, chain_type="stuff", verbose=verbose)
    # response=chain.run(input_documents=docs, question=query)

    # Prompt
    template = """You are an AI trained to be an expert in space mechanism design, analysis, test, and failure investigation.
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Provide a summarized answer, followed up with examples in bullet points.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Chain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT, verbose=True)

    # Run
    response=chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    return response['output_text'], docs