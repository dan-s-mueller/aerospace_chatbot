import os
import glob
import re
import logging
import uuid

import pinecone
import chromadb

import json, jsonlines
from tqdm import tqdm

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as lancghain_Document

from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)
# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

def chunk_docs(docs,
               chunk_method='tiktoken_recursive',
               file=None,
               chunk_size=5000,
               chunk_overlap=0,
               use_json=False):
    docs_out=[]
    if file:
        logging.info('Jsonl file identified: '+file)
    if use_json and os.path.exists(file):
            logging.info('Jsonl file found, using this instead of parsing docs.')
            with open(file, "r") as file_in:
                file_data = [json.loads(line) for line in file_in]
            # Process the file data and put it into the same format as docs_out
            for line in file_data:
                doc_temp = lancghain_Document(page_content=line['text'],
                                              source=line['metadata']['source'],
                                              page=line['metadata']['page'],
                                              metadata=line['metadata'])
                if has_meaningful_content(doc_temp):
                    docs_out.append(doc_temp)
            logging.info('Parsed: '+file)
            logging.info('Number of entries: '+str(len(docs_out)))
            logging.info('Sample entries:')
            logging.info(str(docs_out[0]))
            logging.info(str(docs_out[-1]))
    else:
        logging.info('No jsonl found. Reading and parsing docs.')
        for doc in tqdm(docs,desc='Reading and parsing docs'):
            logging.info('Parsing: '+doc)
            loader = PyPDFLoader(doc)
            data = loader.load_and_split()

            if chunk_method=='tiktoken_recursive':
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                raise NotImplementedError
            pages = text_splitter.split_documents(data)

            # Tidy up text by removing unnecessary characters
            for page in pages:
                page.metadata['source']=os.path.basename(page.metadata['source'])   # Strip path
                page.metadata['page']=int(page.metadata['page'])+1   # Pages are 0 based, update
                page.page_content=re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.page_content)   # Merge hyphenated words
                page.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page.page_content.strip())  # Fix newlines in the middle of sentences
                page.page_content = re.sub(r"\n\s*\n", "\n\n", page.page_content)   # Remove multiple newlines
                doc_temp=lancghain_Document(page_content=page.page_content,
                                            source=page.metadata['source'],
                                            page=page.metadata['page'],
                                            metadata=page.metadata)
                if has_meaningful_content(page):
                    docs_out.append(doc_temp)
            logging.info('Parsed: '+doc)
        if file:
            # Write to a jsonl file, save it.
            with jsonlines.open(file, mode='w') as writer:
                for doc in docs_out: 
                    writer.write(doc.dict())
    return docs_out
def load_docs(index_type,
              docs,
              embeddings_model,
              index_name=None,
              chunk_method='tiktoken_recursive',
              chunk_size=5000,
              chunk_overlap=0,
              clear=False,
              use_json=False,
              file=None,
              batch_size=50):
    """
    Loads PDF documents. If index_name is blank, it will return a list of the data (texts). If it is a name of a pinecone storage, it will return the vector_store.    
    """
    # Chunk docs
    docs_out=chunk_docs(docs,
                        chunk_method=chunk_method,
                        file=file,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_json=use_json)
    # Initialize client
    if index_name:
        if index_type=="Pinecone":
            # Import and initialize Pinecone client
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )
            # Find the existing index, clear for new start
            if clear:
                try:
                    pinecone.describe_index(index_name)
                except:
                    raise Exception(f"Cannot clear index {index_name} because it does not exist.")
                index=pinecone.Index(index_name)
                index.delete(delete_all=True) # Clear the index first, then upload
                logging.info('Cleared database '+index_name)
            # Upsert docs
            try:
                pinecone.describe_index(index_name)
            except:
                logging.info(f"Index {index_name} does not exist. Creating new index.")
                logging.info('Size of embedding used: '+str(embedding_size(embeddings_model)))  # TODO: set this to be backed out of the embedding size
                pinecone.create_index(index_name,dimension=embedding_size(embeddings_model))
                logging.info(f"Index {index_name} created. Adding {len(docs_out)} entries to index.")
                pass
            else:
                logging.info(f"Index {index_name} exists. Adding {len(docs_out)} entries to index.")
            index = pinecone.Index(index_name)
            vectorstore = Pinecone(index, embeddings_model, "page_content") # Set the vector store to calculate embeddings on page_content
            vectorstore = batch_upsert(index_type,
                                       vectorstore,
                                       docs_out,
                                       batch_size=batch_size)
        elif index_type=="ChromaDB":
            # Upsert docs. Defaults to putting this in the ../db directory
            logging.info(f"Creating new index {index_name}.")
            persistent_client = chromadb.PersistentClient(path='../db/chromadb')            
            vectorstore = Chroma(client=persistent_client,
                                 collection_name=index_name,
                                 embedding_function=embeddings_model)
            logging.info(f"Index {index_name} created. Adding {len(docs_out)} entries to index.")
            vectorstore = batch_upsert(index_type,
                                       vectorstore,
                                       docs_out,
                                       batch_size=batch_size)
            logging.info("Documents upserted to f{index_name}.")
            # Test query
            test_query = vectorstore.similarity_search('What are examples of aerosapce adhesives to avoid?')
            logging.info('Test query: '+str(test_query))
            if not test_query:
                raise ValueError("Chroma vector database is not configured properly. Test query failed.")       

        elif index_type=="RAGatouille":
            raise NotImplementedError
    # Return vectorstore or docs
    if index_name:
        return vectorstore
    else:
        return docs_out
def delete_index(index_type,index_name):
    """
    Deletes an existing Pinecone index with the given index_name.
    """
    if index_type=="Pinecone":
        # Import and initialize Pinecone client
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )
        try:
            pinecone.describe_index(index_name)
            logging.info(f"Index {index_name} exists.")
        except:
            raise Exception(f"Index {index_name} does not exist, cannot delete.")
        else:
            pinecone.delete_index(index_name)
            logging.info(f"Index {index_name} deleted.")
    elif index_type=="ChromaDB":
        # Delete existing collection
        logging.info(f"Deleting index {index_name}.")
        persistent_client = chromadb.PersistentClient(path='../db/chromadb')  
        persistent_client.delete_collection(name=index_name)  
        logging.info("Index deleted.")
    elif index_type=="RAGatouille":
            raise NotImplementedError
def batch_upsert(index_type,vectorstore,docs_out,batch_size=50):
    # Batch insert the chunks into the vector store
    for i in range(0, len(docs_out), batch_size):
        chunk_batch = docs_out[i:i + batch_size]
        if index_type=="Pinecone":
            vectorstore.add_documents(chunk_batch)
        elif index_type=="ChromaDB":
            vectorstore.add_documents(chunk_batch)  # Happens to be same for chroma/pinecone, leaving if statement just in case
    return vectorstore
def has_meaningful_content(page):
    """
    Test whether the page has more than 30% words and is more than 5 words.
    """
    text=page.page_content
    num_words = len(text.split())
    alphanumeric_pct = sum(c.isalnum() for c in text) / len(text)
    if num_words < 5 or alphanumeric_pct < 0.3:
        return False
    else:
        return True
def embedding_size(embedding_model):
    """
    Returns the embedding size of the model.
    """
    if isinstance(embedding_model,OpenAIEmbeddings):
        return 1536 # https://platform.openai.com/docs/models/embeddings, test-embedding-ada-002
    elif isinstance(embedding_model,VoyageEmbeddings):
        return 1024 # https://docs.voyageai.com/embeddings/, voyage-02
    else:
        raise NotImplementedError