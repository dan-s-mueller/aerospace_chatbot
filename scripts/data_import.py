"""
@author: dsmueller3760
Script for loading docs into pinecone vector database which can be referenced
"""
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_docs(index_name,
              embeddings_model,
              docs,
              PINECONE_API_KEY=None,
              PINECONE_ENVIRONMENT=None,
              chunk_size=1500,
              chunk_overlap=0,
              clear=True):
    # Import and initialize Pinecone client
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENVIRONMENT') 
    )
    # pinecone.whoami()

    # Find the existing index, clear for new start
    if clear:
        index=pinecone.Index(index_name)
        index.delete(delete_all=True) # Clear the index first, then upload

    for doc in docs:
        print('Parsing: '+doc)
        loader = PyPDFLoader(doc)
        data = loader.load_and_split()
        
        # This is optional, but needed to play with the data parsing.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(data)

        for text in texts:
            text.metadata['page']=text.metadata['page']+1   # Pages are 0 based, update
        
        print('Uploading to pinecone index '+index_name)
        vectorstore = Pinecone.from_documents(texts, embeddings_model, index_name=index_name)