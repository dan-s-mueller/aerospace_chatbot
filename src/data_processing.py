import os
import re
import logging
import shutil
import uuid
import random
from typing import List

from pinecone import Pinecone as pinecone_client
from pinecone import PodSpec
import chromadb

import json, jsonlines

import streamlit as st

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as lancghain_Document

from ragatouille import RAGPretrainedModel

from ragxplorer import RAGxplorer, rag

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)

# Set secrets from environment file
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
VOYAGE_API_KEY=os.getenv('VOYAGE_API_KEY')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN') 

def chunk_docs(docs: List[str],
               rag_type:str='Standard',
               chunk_method:str='tiktoken_recursive',
               file:str=None,
               chunk_size:int=500,
               chunk_overlap:int=0,
               use_json:bool=False,
               show_progress:bool=False):
    """
    Chunk the given list of documents into smaller chunks based on the specified method.

    Args:
        docs (List[str]): List of documents to be chunked.
        chunk_method (str, optional): Method for chunking the documents. Defaults to 'tiktoken_recursive'.
        file (str, optional): Path to the jsonl file. Defaults to None.
        chunk_size (int, optional): Size of each chunk in tokens. Defaults to 500.
        chunk_overlap (int, optional): Number of overlapping tokens between chunks. Defaults to 0.
        use_json (bool, optional): Flag indicating whether to use the jsonl file instead of parsing the docs. Defaults to False.
        show_progress (bool, optional): Flag indicating whether to show progress bar. Defaults to False.

    Returns:
        List[lancghain_Document]: List of chunked documents.
    """

    if show_progress:
        progress_text = "Chunking in progress..."
        my_bar = st.progress(0, text=progress_text)
    chunks=[]
    if file:
        logging.info('Jsonl file to be used: '+file)
    if use_json and os.path.exists(file):
        if rag_type=='Standard':   
            logging.info('Jsonl file found, using this instead of parsing docs.')
            with open(file, "r") as file_in:
                file_data = [json.loads(line) for line in file_in]
            # Process the file data and put it into the same format as chunks
            for i, line in enumerate(file_data):
                doc_temp = lancghain_Document(page_content=line['page_content'],
                                            source=line['metadata']['source'],
                                            page=line['metadata']['page'],
                                            metadata=line['metadata'])
                if has_meaningful_content(doc_temp):
                    chunks.append(doc_temp)
                if show_progress:
                    progress_percentage = i / len(file_data)
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            logging.info('Parsed: '+file)
            logging.info('Number of entries: '+str(len(chunks)))
            logging.info('Sample entries:')
            logging.info(str(chunks[0]))
            logging.info(str(chunks[-1]))
        else:
            raise ValueError("Json import not supported for non-standard RAG types. Please parse the documents (use_json=False).")
    else:
        logging.info('No jsonl found. Reading and parsing docs.')
        logging.info('Chunk size (tokens): '+str(chunk_size))
        logging.info('Chunk overlap (tokens): '+str(chunk_overlap))
        for i, doc in enumerate(docs):
            logging.info('Parsing: '+doc)
            loader = PyPDFLoader(doc)
            page_data = loader.load()

            if rag_type=='Standard':
                if chunk_method=='tiktoken_recursive':
                    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                else:
                    raise NotImplementedError
                page_chunks = text_splitter.split_documents(page_data)

                # Tidy up text by removing unnecessary characters
                for chunk in page_chunks:
                    chunk.metadata['source']=os.path.basename(chunk.metadata['source'])   # Strip path
                    chunk.metadata['page']=int(chunk.metadata['page'])+1   # Pages are 0 based, update
                    chunk.page_content=re.sub(r"(\w+)-\n(\w+)", r"\1\2", chunk.page_content)   # Merge hyphenated words
                    chunk.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", chunk.page_content.strip())  # Fix newlines in the middle of sentences
                    chunk.page_content = re.sub(r"\n\s*\n", "\n\n", chunk.page_content)   # Remove multiple newlines
                    # Add metadata to the end of the page content, some RAG models don't have metadata.
                    chunk.page_content += str(chunk.metadata)
                    chunk_temp=lancghain_Document(page_content=chunk.page_content,
                                                source=chunk.metadata['source'],
                                                page=chunk.metadata['page'],
                                                metadata=chunk.metadata)
                    if has_meaningful_content(chunk):
                        chunks.append(chunk_temp)
            elif rag_type=='Parent Child'
            if show_progress:
                progress_percentage = i / len(docs)
                my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
        logging.info('Parsed: '+doc)
        logging.info('Sample entries:')
        logging.info(str(chunks[0]))
        logging.info(str(chunks[-1]))
        if file:
            # Write to a jsonl file, save it.
            logging.info('Writing to jsonl file: '+file)
            with jsonlines.open(file, mode='w') as writer:
                for doc in chunks: 
                    writer.write(doc.dict())
            logging.info('Written: '+file)
        if show_progress:
            my_bar.empty()
    return chunks
def load_docs(index_type,
              docs,
              query_model,
              rag_type='Standard',
              index_name=None,
              chunk_method='tiktoken_recursive',
              chunk_size=500,
              chunk_overlap=0,
              clear=False,
              use_json=False,
              file=None,
              batch_size=50,
              local_db_path='../db',
              show_progress=False):
    """
    Load documents into an index for a given index type.

    Args:
        index_type (str): The type of index to use (e.g., "Pinecone", "ChromaDB", "RAGatouille").
        docs (list): The list of documents to load into the index.
        query_model (str): The query model to use for embedding calculation.
        index_name (str, optional): The name of the index. Defaults to None.
        chunk_method (str, optional): The method for chunking the documents. Defaults to 'tiktoken_recursive'.
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 0.
        clear (bool, optional): Whether to clear the index before loading documents. Defaults to False.
        use_json (bool, optional): Whether to use JSON format for documents. Defaults to False.
        file (str, optional): The file to load documents from. Defaults to None.
        batch_size (int, optional): The batch size for upserting documents. Defaults to 50.
        local_db_path (str, optional): The path to the local database. Defaults to '../db'.
        show_progress (bool, optional): Whether to show progress during upsert. Defaults to False.

    Returns:
        vectorstore (object): The vector store containing the loaded documents, if index_name is provided.
        docs_out (list): The chunked documents, if index_name is not provided.
    """

    # Chunk docs
    if rag_type!='Standard':
        # TODO: https://colab.research.google.com/github/datastax/ragstack-ai/blob/main/examples/notebooks/advancedRAG.ipynb
        raise NotImplementedError
    else:
        docs_out=chunk_docs(docs,
                            rag_type='Standard',
                            chunk_method=chunk_method,
                            file=file,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            use_json=use_json)
    # Initialize client
    if index_name:
        if index_type=="Pinecone":
            if clear:
                delete_index(index_type,index_name,local_db_path=local_db_path)
            # Upsert docs
            logging.info(f"Creating new index {index_name}.")
            pc = pinecone_client(api_key=PINECONE_API_KEY)
            try:
                pc.describe_index(index_name)
            except:
                logging.info(f"Index {index_name} does not exist. Creating new index.")
                logging.info('Size of embedding used: '+str(embedding_size(query_model)))  # TODO: set this to be backed out of the embedding size
                pc.create_index(index_name,
                                dimension=embedding_size(query_model),
                                metric="cosine",
                                spec=PodSpec(environment="us-west1-gcp", pod_type="p1.x1"))
                logging.info(f"Index {index_name} created. Adding {len(docs_out)} entries to index.")
                pass
            else:
                logging.info(f"Index {index_name} exists. Adding {len(docs_out)} entries to index.")
            index = pc.Index(index_name)
            vectorstore = Pinecone(index, query_model, "page_content") # Set the vector store to calculate embeddings on page_content
            if rag_type!='Standard':
                # TODO: https://colab.research.google.com/github/datastax/ragstack-ai/blob/main/examples/notebooks/advancedRAG.ipynb
                raise NotImplementedError
            vectorstore = batch_upsert(index_type,
                                       vectorstore,
                                       docs_out,
                                       batch_size=batch_size,
                                       show_progress=show_progress)
        elif index_type=="ChromaDB":
            if clear:
                delete_index(index_type,index_name,local_db_path=local_db_path)
            # Upsert docs. Defaults to putting this in the local_db_path directory
            logging.info(f"Creating new index {index_name}.")
            logging.info(f"Local database path: {local_db_path+'/chromadb'}")
            persistent_client = chromadb.PersistentClient(path=local_db_path+'/chromadb')            
            vectorstore = Chroma(client=persistent_client,
                                 collection_name=index_name,
                                 embedding_function=query_model)
            
            if rag_type!='Standard':
                # TODO: https://colab.research.google.com/github/datastax/ragstack-ai/blob/main/examples/notebooks/advancedRAG.ipynb
                raise NotImplementedError
            vectorstore = batch_upsert(index_type,
                                       vectorstore,
                                       docs_out,
                                       batch_size=batch_size,
                                       show_progress=show_progress)
            logging.info("Documents upserted to f{index_name}.")
            # Test query
            test_query = vectorstore.similarity_search('What are examples of aerosapce adhesives to avoid?')
            logging.info('Test query: '+str(test_query))
            if not test_query:
                raise ValueError("Chroma vector database is not configured properly. Test query failed.")       
        elif index_type=="RAGatouille":
            if clear:
                delete_index(index_type,index_name,local_db_path=local_db_path)
            logging.info(f'Setting up RAGatouille model {query_model}')
            vectorstore = RAGPretrainedModel.from_pretrained(query_model)
            logging.info('RAGatouille model set: '+str(vectorstore))

            if rag_type!='Standard':
                # TODO: https://colab.research.google.com/github/datastax/ragstack-ai/blob/main/examples/notebooks/advancedRAG.ipynb
                raise NotImplementedError
            
            # Create an index from the vectorstore.
            docs_out_colbert = [doc.page_content for doc in docs_out]
            if chunk_size>500:
                raise ValueError("RAGatouille cannot handle chunks larger than 500 tokens. Reduce token count.")
            vectorstore.index(
                collection=docs_out_colbert,
                index_name=index_name,
                max_document_length=chunk_size,
                overwrite_index=True,
                split_documents=True,
            )
            logging.info(f"Index created: {vectorstore}")

            # Move the directory to the db folder
            logging.info(f"Moving RAGatouille index to {local_db_path}")
            ragatouille_path = os.path.join(local_db_path, '.ragatouille')
            if os.path.exists(ragatouille_path):
                shutil.rmtree(ragatouille_path)
                logging.info(f"RAGatouille index deleted from {ragatouille_path}")
            shutil.move('./.ragatouille', local_db_path)
            logging.info(f"RAGatouille index created in {local_db_path}:"+str(vectorstore))

    # Return vectorstore or docs
    if index_name:
        return vectorstore
    else:
        return docs_out
def delete_index(index_type: str, index_name: str, local_db_path: str = '../db'):
    """
    Delete an index based on the specified index type and name.

    Args:
        index_type (str): The type of index to delete. Valid options are "Pinecone", "ChromaDB", and "RAGatouille".
        index_name (str): The name of the index to delete.
        local_db_path (str, optional): The path to the local database. Defaults to '../db'.

    Raises:
        Exception: If the specified index does not exist.

    Returns:
        None
    """
    if index_type == "Pinecone":
        pc = pinecone_client(api_key=PINECONE_API_KEY)
        try:
            pc.describe_index(index_name)
        except:
            raise Exception(f"Cannot clear index {index_name} because it does not exist.")
        index = pc.Index(index_name)
        index.delete(delete_all=True)  # Clear the index first, then upload
        logging.info('Cleared database ' + index_name)
    elif index_type == "ChromaDB":
        try:
            persistent_client = chromadb.PersistentClient(path=local_db_path + '/chromadb')
            indices = persistent_client.list_collections()
            logging.info('Available databases: ' + str(indices))
            for idx in indices:
                if index_name in idx.name:
                    logging.info(f"Clearing index {idx.name}...")
                    persistent_client.delete_collection(name=idx.name)
                    logging.info(f"Index {idx.name} cleared.")
        except:
            raise Exception(f"Cannot clear index {index_name} because it does not exist.")
        logging.info('Cleared database and matching databases ' + index_name)
    elif index_type == "RAGatouille":
        try:
            ragatouille_path = os.path.join(local_db_path, '.ragatouille')
            shutil.rmtree(ragatouille_path)
        except:
            raise Exception(f"Cannot clear index {index_name} because it does not exist.")
from typing import List

def batch_upsert(index_type: str, vectorstore: any, docs_out: List, batch_size: int = 50, show_progress: bool = False):
    """
    Upserts a batch of documents into a vector store.

    Args:
        index_type (str): The type of vector store index ("Pinecone" or "ChromaDB").
        vectorstore (any): The vector store object.
        docs_out (List): The list of documents to upsert.
        batch_size (int, optional): The size of each batch. Defaults to 50.
        show_progress (bool, optional): Whether to show progress bar. Defaults to False.

    Returns:
        any: The updated vector store object.
    """
    if show_progress:
        progress_text = "Upsert in progress..."
    my_bar = st.progress(0, text=progress_text)
    for i in range(0, len(docs_out), batch_size):
        chunk_batch = docs_out[i:i + batch_size]
        if index_type == "Pinecone":
            vectorstore.add_documents(chunk_batch)
        elif index_type == "ChromaDB":
            vectorstore.add_documents(chunk_batch)  # Happens to be same for chroma/pinecone, leaving if statement just in case
        if show_progress:
            progress_percentage = i / len(docs_out)
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    if show_progress:
        my_bar.empty()
    return vectorstore
def has_meaningful_content(page):
    """
    Check if a page has meaningful content.

    Args:
        page (Page): The page object to check.

    Returns:
        bool: True if the page has meaningful content, False otherwise.
    """
    text = page.page_content
    num_words = len(text.split())
    alphanumeric_pct = sum(c.isalnum() for c in text) / len(text)
    if num_words < 5 or alphanumeric_pct < 0.3:
        return False
    else:
        return True
def embedding_size(embedding_model:any):
    """
    Returns the size of the embedding for a given embedding model.

    Args:
        embedding_model (object): The embedding model to get the size for.

    Returns:
        int: The size of the embedding.

    Raises:
        NotImplementedError: If the embedding model is not supported.
    """
    if isinstance(embedding_model,OpenAIEmbeddings):
        return 1536 # https://platform.openai.com/docs/models/embeddings, test-embedding-ada-002
    elif isinstance(embedding_model,VoyageEmbeddings):
        return 1024 # https://docs.voyageai.com/embeddings/, voyage-02
    else:
        raise NotImplementedError
def process_chunk(json_file:str,
                  llm:any,
                  clean_data:bool=False,
                  tag_data:bool=False,
                  question_data:bool=False):
    docs_out=[]
    with open(json_file, "r") as file_in:
        file_data = [json.loads(line) for line in file_in]
        # Process the file data and put it into the same format as docs_out
        for line in file_data:
            doc_temp = lancghain_Document(page_content=line['page_content'],
                                            source=line['metadata']['source'],
                                            page=line['metadata']['page'],
                                            metadata=line['metadata'])
            docs_out.append(doc_temp)
    # TODO: write out this function
    # clean data: use cheap llm to clean data
    # tag data: use llm to tag data and add metadata for filtering/grouping later
    # question data: use llm to generate questions from data
    
def reduce_vector_query_size(rx_client:RAGxplorer,chroma_client:chromadb,vector_qty:int,verbose:bool=False):
    """Reduce the number of vectors in the RAGxplorer client's vector database.

    Args:
        rx_client (RAGxplorer): The RAGxplorer client object.
        chroma_client (chromadb): The chromadb client object.
        vector_qty (int): The desired number of vectors to keep.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        RAGxplorer: The updated RAGxplorer client object.
    """
    ids = rx_client._vectordb.get()['ids']
    embeddings = rag.get_doc_embeddings(rx_client._vectordb)
    text = rag.get_docs(rx_client._vectordb)

    if verbose:
        print(' ~ Reducing the number of vectors from '+str(len(embeddings))+' to '+str(vector_qty)+'...')
    indices = random.sample(range(len(embeddings)), vector_qty)
    id = str(uuid.uuid4())[:8]
    temp_index_name=rx_client._vectordb.name+'-'+id
    
    # Create a temporary index with the reduced number of vectors
    chroma_client.create_collection(name=temp_index_name,embedding_function=rx_client._chosen_embedding_model)
    temp_collection = chroma_client.get_collection(name=temp_index_name,embedding_function=rx_client._chosen_embedding_model)
    temp_collection.add(
        ids=[ids[i] for i in indices],
        embeddings=[embeddings[i] for i in indices],
        documents=[text[i] for i in indices]
    )

    # Replace the original index with the temporary one
    rx_client._vectordb = temp_collection
    rx_client._documents.embeddings = rag.get_doc_embeddings(rx_client._vectordb)
    rx_client._documents.text = rag.get_docs(rx_client._vectordb)
    rx_client._documents.ids = rx_client._vectordb.get()['ids']
    if verbose:
        print('Reduced number of vectors to '+str(len(rx_client._documents.embeddings))+' ✓')
        print('Copy of database saved as '+temp_index_name+' ✓')
    return rx_client

def export_data_viz(rx_client:RAGxplorer,df_export_path:str):
    """Export visualization data and UMAP parameters of RAGxplorer object to a JSON file.

    Args:
        rx_client (RAGxplorer): The RAGxplorer object containing the visualization data.
        df_export_path (str): The file path to export the JSON data.

    """
    export_data = {'visualization_index_name' : rx_client._vectordb.name,
                   'umap_params': rx_client._projector.get_params(),
                   'viz_data': rx_client._VizData.base_df.to_json(orient='split')}

    # Save the data to a JSON file
    with open(df_export_path, 'w') as f:
        json.dump(export_data, f, indent=4)