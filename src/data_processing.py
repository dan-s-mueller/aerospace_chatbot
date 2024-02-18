from prompts import TEST_QUERY_PROMPT

import os
import re
import logging
import shutil
import uuid
import random
from pathlib import Path
from typing import List

from pinecone import Pinecone as pinecone_client
from pinecone import PodSpec
import chromadb

import json, jsonlines

import streamlit as st

from langchain_pinecone import Pinecone
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

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

def load_docs(index_type,
              docs,
              query_model,
              rag_type='Standard',
              index_name=None,
              chunk_method='character_recursive',
              chunk_size=500,
              chunk_overlap=0,
              clear=False,
              file_out=None,
              batch_size=50,
              local_db_path='../db',
              show_progress=False):
    """
    Load documents into the specified index.

    Args:
        index_type (str): The type of index to use.
        docs (list): The list of documents to load.
        query_model (str): The query model to use.
        rag_type (str, optional): The type of RAG to use. Defaults to 'Standard'.
        index_name (str, optional): The name of the index. Defaults to None.
        chunk_method (str, optional): The method to chunk the documents. Defaults to 'tiktoken_recursive'.
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 0.
        clear (bool, optional): Whether to clear the index before loading documents. Defaults to False.
        file_out (str, optional): The output file path. Defaults to None.
        batch_size (int, optional): The batch size for upserting documents. Defaults to 50.
        local_db_path (str, optional): The local database path. Defaults to '../db'.
        show_progress (bool, optional): Whether to show progress during loading. Defaults to False.

    Returns:
        vectorstore: The updated vectorstore.
    """
    # Check for illegal things
    if not clear and (rag_type == 'Parent-Child' or rag_type == 'Summary'):
        raise ValueError('Parent-Child databases must be cleared before loading new documents.')

    # Chunk docs
    chunker=chunk_docs(docs,
                       rag_type=rag_type,
                       chunk_method=chunk_method,
                       chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap,
                       file_out=file_out,
                       show_progress=True)
        
    # Set index names for special databases
    if rag_type == 'Parent-Child':
        index_name = index_name + '-parent-child'
    if rag_type == 'Summary':
        index_name = index_name + '-summary'

    # Initialize client an upsert docs
    vectorstore = initialize_database(index_type, 
                                      index_name, 
                                      query_model, 
                                      clear=clear, 
                                      local_db_path=local_db_path,
                                      init_ragatouille=True)
    vectorstore, retriever = upsert_docs(index_type,
                                         index_name,
                                         vectorstore,
                                         chunker,
                                         batch_size=batch_size,
                                         show_progress=show_progress,
                                         local_db_path=local_db_path)
    logging.info(f"Documents upserted to {index_name}.")
    return vectorstore
def chunk_docs(docs: List[str],
               rag_type:str='Standard',
               chunk_method:str='character_recursive',
               file_out:str=None,
               chunk_size:int=500,
               chunk_overlap:int=0,
               k_parent:int=4,
               show_progress:bool=False):
    if show_progress:
        progress_text = "Chunking in progress..."
        my_bar = st.progress(0, text=progress_text)
    pages=[]
    chunks=[]
    logging.info('No jsonl found. Reading and parsing docs.')
    logging.info('Chunk size (tokens): '+str(chunk_size))
    logging.info('Chunk overlap (tokens): '+str(chunk_overlap))

    # Parse doc pages
    for i, doc in enumerate(docs):
        logging.info('Parsing: '+doc)
        loader = PyPDFLoader(doc)
        page_data = loader.load()

        # Clean up page info, update some metadata
        for page in page_data:
            page.metadata['source']=os.path.basename(page.metadata['source'])   # Strip path
            page.metadata['page']=int(page.metadata['page'])+1   # Pages are 0 based, update
            page.page_content=re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.page_content)   # Merge hyphenated words
            page.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page.page_content.strip())  # Fix newlines in the middle of sentences
            page.page_content = re.sub(r"\n\s*\n", "\n\n", page.page_content)   # Remove multiple newlines
            if has_meaningful_content(page):
                pages.append(page)
        if show_progress:
            progress_percentage = i / len(docs)
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    
    # Process pages
    if rag_type!='Parent-Child': 
        if chunk_method=='character_recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise NotImplementedError
        page_chunks = text_splitter.split_documents(pages)

        for chunk in page_chunks:
            chunk.page_content += str(chunk.metadata)    # Add metadata to the end of the page content, some RAG models don't have metadata.
            if has_meaningful_content(chunk):
                chunks.append(chunk)
        logging.info('Parsed: '+doc)
        logging.info('Sample entries:')
        logging.info(str(chunks[0]))
        logging.info(str(chunks[-1]))
        if file_out:
            # Write to a jsonl file, save it.
            logging.info('Writing to jsonl file: '+file_out)
            with jsonlines.open(file_out, mode='w') as writer:
                for doc in chunks: 
                    writer.write(doc.dict())
            logging.info('Written: '+file_out)
        if show_progress:
            my_bar.empty()
        return {'rag':'Standard',
                'pages':pages,
                'chunks':chunks,
                'splitters':text_splitter}
    elif rag_type=='Parent-Child': 
        if chunk_method=='character_recursive':
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size*k_parent, chunk_overlap=chunk_overlap)
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise NotImplementedError
        
        # Split up parent chunks
        parent_chunks = parent_splitter.split_documents(pages)

        doc_ids = [str(uuid.uuid4()) for _ in parent_chunks]
        
        # Split up child chunks
        id_key = "doc_id"
        chunks = []
        for i, doc in enumerate(parent_chunks):
            _id = doc_ids[i]
            _chunks = child_splitter.split_documents([doc])
            for _doc in _chunks:
                _doc.metadata[id_key] = _id
            chunks.extend(_chunks)

        if show_progress:
            my_bar.empty()
        return {'rag':'Parent-Child',
                'pages':{'doc_ids':doc_ids,'parent_chunks':parent_chunks},
                'chunks':chunks,
                'splitters':{'parent_splitter':parent_splitter,'child_splitter':child_splitter}}
    elif rag_type=='Summary':
        raise NotImplementedError
def initialize_database(index_type: str, 
                        index_name: str, 
                        query_model: str, 
                        local_db_path: str = None, 
                        clear: bool = False,
                        test_query: bool = False,
                        init_ragatouille: bool = False):
    """
    Initializes the database based on the specified index type.

    Args:
        index_type (str): The type of index to use (e.g., "Pinecone", "ChromaDB", "RAGatouille").
        index_name (str): The name of the index.
        query_model (str): The query model to use.
        local_db_path (str, optional): The local database path for ChromaDB. Defaults to None.
        clear (bool, optional): Whether to clear the index before creating a new one. Defaults to False.
        test_query (bool, optional): Whether to perform a test query. Defaults to False.
        init_ragatouille (bool, optional): Whether to initialize the RAGatouille model. Defaults to False.

    Returns:
        vectorstore: The initialized vector store.

    Raises:
        Exception: If the vector database is not configured properly.
        ValueError: If the vector database or llm is not configured properly.
    """
    if index_type == "Pinecone":
        if clear:
            delete_index(index_type, index_name)
        logging.info(f"Creating new index {index_name}.")
        pc = pinecone_client(api_key=PINECONE_API_KEY)
        
        try:
            pc.describe_index(index_name)
        except:
            logging.info(f"Index {index_name} does not exist. Creating new index.")
            pc.create_index(index_name,
                            dimension=embedding_size(query_model),
                            metric="cosine",
                            spec=PodSpec(environment="us-west1-gcp", pod_type="p1.x1"))
            logging.info(f"Index {index_name} created.")
        else:
            logging.info(f"Index {index_name} exists.")
        
        index = pc.Index(index_name)
        vectorstore = Pinecone(index, query_model, "page_content")  # Set the vector store to calculate embeddings on page_content

    elif index_type == "ChromaDB":
        if clear:
            delete_index(index_type, index_name, local_db_path=local_db_path)
        # Upsert docs. Defaults to putting this in the local_db_path directory
        logging.info(f"Creating new index {index_name}.")
        logging.info(f"Local database path: {local_db_path+'/chromadb'}")
        persistent_client = chromadb.PersistentClient(path=local_db_path+'/chromadb')            
        vectorstore = Chroma(client=persistent_client,
                                collection_name=index_name,
                                embedding_function=query_model)        

    elif index_type == "RAGatouille":
        if clear:
            delete_index(index_type, index_name, local_db_path=local_db_path)
        logging.info(f'Setting up RAGatouille model {query_model}')
        if init_ragatouille:    # Used if the index is not already set
            vectorstore = RAGPretrainedModel.from_pretrained(query_model)
        else:   # Used if the index is already set
            vectorstore = query_model    # The index is picked up directly.
        logging.info('RAGatouille model set: ' + str(vectorstore))

    if test_query:
        try:    # Test query
            test_query = vectorstore.similarity_search(TEST_QUERY_PROMPT)
        except:
            raise Exception("Vector database is not configured properly. Test query failed. Likely the index does not exist.")
        logging.info('Test query: ' + str(test_query))
        if not test_query:
            raise ValueError("Vector database or llm is not configured properly. Test query failed.")
        else:
            logging.info('Test query succeeded!')

    return vectorstore
def upsert_docs(index_type:str, 
                index_name:str,
                vectorstore:any, 
                chunker:dict, 
                batch_size:int = 50, 
                show_progress:bool = False,
                local_db_path:str = '../db'):
    """
    Upserts documents into the specified index.

    Args:
        index_type (str): The type of index to upsert the documents into.
        index_name (str): The name of the index.
        vectorstore (any): The vectorstore object used for storing vectors.
        chunker (dict): The chunker object containing the chunks of documents.
        batch_size (int, optional): The batch size for upserting documents. Defaults to 50.
        show_progress (bool, optional): Whether to show progress during the upsert process. Defaults to False.
        local_db_path (str, optional): The local path to the database folder. Defaults to '../db'.

    Returns:
        tuple: A tuple containing the updated vectorstore and retriever objects.
    """
    if show_progress:
        progress_text = "Upsert in progress..."
    my_bar = st.progress(0, text=progress_text)

    if chunker['rag']=='Standard':
        # Upsert each chunk in batches
        if index_type != "RAGatouille":
            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                vectorstore.add_documents(chunk_batch)  # Happens to be same for chroma/pinecone/others
                if show_progress:
                    progress_percentage = i / len(chunker['chunks'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            logging.info(f"Index created: {vectorstore}")
        elif index_type == "RAGatouille":
            logging.info(f"Creating index {index_name} from RAGatouille.")
            # Create an index from the vectorstore.
            vectorstore.index(
                collection=[chunk.page_content for chunk in chunker['chunks']],
                index_name=index_name,
                max_document_length=chunker['splitters']._chunk_size,
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
        retriever=vectorstore.as_retriever()
    elif chunker['rag']=='Parent-Child':
        if index_type != "RAGatouille":
            # TODO: make sure this is deleted/udpated whenever a parent-child database is udpated.
            root_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
            store = LocalFileStore(root_path)
            
            id_key = "doc_id"
            retriever = MultiVectorRetriever(vectorstore=vectorstore,byte_store=store,id_key=id_key)

            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                retriever.vectorstore.add_documents(chunk_batch)
                if show_progress:
                    progress_percentage = i / len(chunker['chunks'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            
            # Index parent docs all at once
            retriever.docstore.mset(list(zip(chunker['pages']['doc_ids'], 
                                             chunker['pages']['parent_chunks'])))
            logging.info(f"Index created: {vectorstore}")
        elif index_type == "RAGatouille":
            raise Exception('RAGAtouille only supports standard RAG.')
    elif chunker['rag']=='Summary':
        raise NotImplementedError
    if show_progress:
        my_bar.empty()
    return vectorstore, retriever
def delete_index(index_type: str, 
                 index_name: str, 
                 rag_type: str,
                 local_db_path: str = '../db'):
    """
    Deletes an index based on the specified index type and index name.

    Args:
        index_type (str): The type of index to delete. Possible values are "Pinecone", "ChromaDB", or "RAGatouille".
        index_name (str): The name of the index to delete.
        local_db_path (str, optional): The path to the local database. Defaults to '../db'.

    Raises:
        Exception: If the index does not exist.

    Returns:
        None
    """
    if rag_type != 'Parent-Child':
        if index_type == "Pinecone":
            pc = pinecone_client(api_key=PINECONE_API_KEY)
            try:
                pc.describe_index(index_name)
            except:
                raise Exception(f"Cannot clear index {index_name} because it does not exist. Create the index first.")
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
    if rag_type == 'Parent-Child' or rag_type == 'Summary':
        raise NotImplementedError
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
    if len(text)==0:
        return False
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

def process_chunk(json_file:str,
                  llm:any,
                  clean_data:bool=False,
                  tag_data:bool=False,
                  question_data:bool=False):
    
    # TODO: write out this function
    # docs_out=[]
    # with open(json_file, "r") as file_in:
    #     file_data = [json.loads(line) for line in file_in]
    #     # Process the file data and put it into the same format as docs_out
    #     for line in file_data:
    #         doc_temp = lancghain_Document(page_content=line['page_content'],
    #                                         source=line['metadata']['source'],
    #                                         page=line['metadata']['page'],
    #                                         metadata=line['metadata'])
    #         docs_out.append(doc_temp)

    # clean data: use cheap llm to clean data
    # tag data: use llm to tag data and add metadata for filtering/grouping later
    # question data: use llm to generate questions from data
    return None