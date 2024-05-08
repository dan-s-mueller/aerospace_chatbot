from prompts import SUMMARIZE_TEXT

import os, re, shutil
import hashlib
from pathlib import Path
from typing import List, Union

from tenacity import retry, stop_after_attempt, wait_exponential

from pinecone import Pinecone as pinecone_client
from pinecone import PodSpec

import chromadb
from chromadb import PersistentClient

import json, jsonlines

import streamlit as st

from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ragatouille import RAGPretrainedModel

from renumics import spotlight
from renumics.spotlight import dtypes as spotlight_dtypes

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from datasets import Dataset

import admin
from prompts import CLUSTER_LABEL

def load_docs(index_type:str,
              docs:List[str],
              query_model:object,
              rag_type:str='Standard',
              index_name:str=None,
              n_merge_pages:int=None,
              chunk_method:str='character_recursive',
              chunk_size:int=500,
              chunk_overlap:int=0,
              clear:bool=False,
              file_out:str=None,
              batch_size:int=50,
              local_db_path:str='.',
              llm=None,
              show_progress:bool=False):
    """
    Loads documents into the specified index.

    Args:
        index_type (str): The type of index to use.
        docs: The documents to load.
        query_model (object): The query model to use.
        rag_type (str, optional): The type of RAG (Retrieval-Augmented Generation) to use. Defaults to 'Standard'.
        index_name (str, optional): The name of the index. Defaults to None.
        n_merge_pages (int, optional): Number of pages to to merge when loading. Defaults to 0.
        chunk_method (str, optional): The method to chunk the documents. Defaults to 'character_recursive'.
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 0.
        clear (bool, optional): Whether to clear the index before loading new documents. Defaults to False.
        file_out (str, optional): The output file path. Defaults to None.
        batch_size (int, optional): The batch size for upserting documents. Defaults to 50.
        local_db_path (str, optional): The local database path. Defaults to '../../db'.
        llm (optional): The language model to use. Defaults to None.
        show_progress (bool, optional): Whether to show progress during the loading process. Defaults to False.

    Returns:
        vectorstore: The updated vectorstore.
    """
    # Check for illegal things
    if not clear and (rag_type == 'Parent-Child' or rag_type == 'Summary'):
        raise ValueError('Parent-Child databases must be cleared before loading new documents.')

    # Chunk docs
    chunker=chunk_docs(docs,
                       rag_type=rag_type,
                       n_merge_pages=n_merge_pages,
                       chunk_method=chunk_method,
                       chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap,
                       file_out=file_out,
                       llm=llm,
                       show_progress=show_progress)
        
    # Set index names for special databases
    if rag_type == 'Parent-Child':
        index_name = index_name + '-parent-child'
    if rag_type == 'Summary':
        index_name = index_name + llm.model_name.replace('/', '-') + '-summary' 

    # Initialize client an upsert docs
    vectorstore = initialize_database(index_type, 
                                      index_name, 
                                      query_model,
                                      rag_type=rag_type,
                                      clear=clear, 
                                      local_db_path=local_db_path,
                                      init_ragatouille=True,
                                      show_progress=show_progress)
    vectorstore, _ = upsert_docs(index_type,
                                 index_name,
                                 vectorstore,
                                 chunker,
                                 batch_size=batch_size,
                                 show_progress=show_progress,
                                 local_db_path=local_db_path)
    return vectorstore
def chunk_docs(docs: List[str],
               rag_type:str='Standard',
               chunk_method:str='character_recursive',
               file_out:str=None,
               n_merge_pages:int=None,
               chunk_size:int=500,
               chunk_overlap:int=0,
               k_parent:int=4,
               llm=None,
               show_progress:bool=False):
    """
    Chunk the given list of documents into smaller chunks based on the specified parameters.

    Args:
        docs (List[str]): The list of document paths to be chunked.
        rag_type (str, optional): The type of chunking method to be used. Defaults to 'Standard'.
        chunk_method (str, optional): The method of chunking to be used. Defaults to 'character_recursive'. None will take whole PDF pages as documents.
        file_out (str, optional): The output file path to save the chunked documents. Defaults to None.
        n_merge_pages (int, optional): Number of pages to to merge when loading. Defaults to None.
        chunk_size (int, optional): The size of each chunk in tokens. Defaults to 500. Only used if chunk_method is not None.
        chunk_overlap (int, optional): The overlap between chunks in tokens. Defaults to 0. Only used if chunk_method is not None.
        k_parent (int, optional): The number of parent chunks to split into child chunks for 'Parent-Child' rag_type. Defaults to 4.
        llm (None, optional): The language model to be used for generating summaries. Defaults to None.
        show_progress (bool, optional): Whether to show the progress bar during chunking. Defaults to False.

    Returns:
        dict: A dictionary containing the chunking results based on the specified rag_type.
    """
    if show_progress:
        progress_text = "Chunking in progress..."
        my_bar = st.progress(0, text=progress_text)
    pages=[]
    chunks=[]
    
    # Parse doc pages
    for i, doc in enumerate(docs):
        
        loader = PyPDFLoader(doc)
        doc_page_data = loader.load()

        # Clean up page info, update some metadata
        doc_pages=[]
        for doc_page in doc_page_data:
            doc_page=_sanitize_raw_page_data(doc_page)
            if doc_page is not None:
                doc_pages.append(doc_page)
        if show_progress:
            progress_percentage = i / len(docs)
            my_bar.progress(progress_percentage, text=f'Reading documents...{progress_percentage*100:.2f}%')
        
        # Merge pages if option is selected
        if n_merge_pages:
            for i in range(0, len(doc_pages), n_merge_pages):
                group = doc_pages[i:i+n_merge_pages]
                group_page_content=' '.join([doc.page_content for doc in group])
                group_metadata = {'page': str([doc.metadata['page'] for doc in group]), 
                                  'source': str([doc.metadata['source'] for doc in group])}
                merged_doc = Document(page_content=group_page_content, metadata=group_metadata)
                pages.append(merged_doc)
        else:
            pages.extend(doc_pages)
    
    # Process pages
    if rag_type=='Standard': 
        if chunk_method=='character_recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                           chunk_overlap=chunk_overlap,
                                                           add_start_index=True)
            page_chunks = text_splitter.split_documents(pages)
            for i, chunk in enumerate(page_chunks):
                if show_progress:
                    progress_percentage = i / len(page_chunks)
                    my_bar.progress(progress_percentage, text=f'Chunking documents...{progress_percentage*100:.2f}%')
                chunks.append(chunk)    # Not sanitized because the page already was
        elif chunk_method=='None':
            text_splitter = None
            chunks = pages  # No chunking, take whole pages as documents
        else:
            raise NotImplementedError
        
        if file_out:
            # Write to a jsonl file, save it.
            chunk_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunks]   # add ID which is the hash of metadata
            with jsonlines.open(file_out, mode='w') as writer:
                for doc, chunk_id in zip(chunks, chunk_ids): 
                    doc_out=doc.dict()
                    doc_out['chunk_id'] = chunk_id  # Add chunk_id to the jsonl file
                    writer.write(doc_out)
        if show_progress:
            my_bar.empty()
        return {'rag':'Standard',
                'pages':pages,
                'chunks':chunks, 
                'splitters':text_splitter}
    elif rag_type=='Parent-Child': 
        if chunk_method=='character_recursive':
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size*k_parent, 
                                                           chunk_overlap=chunk_overlap,
                                                           add_start_index=True)    # Without add_start_index, will not be a unique id
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                          chunk_overlap=chunk_overlap,
                                                          add_start_index=True)
        elif chunk_method=='None':
            raise ValueError("You must specify a chunk_method with rag_type=Parent-Child.")
        else:
            raise NotImplementedError
        
        # Split up parent chunks
        parent_chunks = parent_splitter.split_documents(pages)
        doc_ids = [str(_stable_hash_meta(parent_chunk.metadata)) for parent_chunk in parent_chunks]
        
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
    elif rag_type == 'Summary':
        if show_progress:
            my_bar.empty()
            my_bar = st.progress(0, text='Generating summaries...')

        id_key = "doc_id"
        doc_ids = [str(_stable_hash_meta(page.metadata)) for page in pages]
        chain = (
            {"doc": lambda x: x.page_content}
            | SUMMARIZE_TEXT
            | llm
            | StrOutputParser()
        )

        summaries = []
        for i, page in enumerate(pages):
            summary = chain.invoke(page)
            summaries.append(summary)
            if show_progress:
                progress_percentage = i / len(pages)
                my_bar.progress(progress_percentage, text=f'Generating summaries...{progress_percentage*100:.2f}%')

        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        if show_progress:
            my_bar.empty()
        return {'rag':'Summary',
                'pages':{'doc_ids':doc_ids,'docs':pages},
                'summaries':summary_docs,
                'llm':llm}
    else:
        raise NotImplementedError
def initialize_database(index_type: str, 
                        index_name: str, 
                        query_model: object,
                        rag_type: str,
                        local_db_path: str = None, 
                        clear: bool = False,
                        init_ragatouille: bool = False,
                        show_progress: bool = False):
    """Initializes the database based on the specified parameters.

    Args:
        index_type (str): The type of index to use (e.g., "Pinecone", "ChromaDB", "RAGatouille").
        index_name (str): The name of the index.
        query_model (object): The query model to use.
        rag_type (str): The type of RAG model to use.
        local_db_path (str, optional): The path to the local database. Defaults to None.
        clear (bool, optional): Whether to clear the index. Defaults to False.
        init_ragatouille (bool, optional): Whether to initialize the RAGatouille model. Defaults to False.
        show_progress (bool, optional): Whether to show the progress bar. Defaults to False.

    Returns:
        vectorstore: The initialized vector store.

    Raises:
        NotImplementedError: If the specified index type is not implemented.
    """

    if show_progress:
        progress_text = "Database initialization..."
        my_bar = st.progress(0, text=progress_text)

    if index_type == "Pinecone":

        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        
        try:
            pc.describe_index(index_name)
        except:
            pc.create_index(index_name,
                            dimension=_embedding_size(query_model),
                            spec=PodSpec(environment="us-west1-gcp", pod_type="p1.x1"))
        
        index = pc.Index(index_name)
        vectorstore=PineconeVectorStore(index,
                                        index_name=index_name, 
                                        embedding=query_model,
                                        text_key='page_content',
                                        pinecone_api_key=os.getenv('PINECONE_API_KEY'))
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    elif index_type == "ChromaDB":
        # TODO add in collection metadata like this:
        #         collection_metadata = {}
        # if embeddings_model is not None:
        #     model_name, model_type = get_embeddings_model_config(embeddings_model)
        #     collection_metadata["model_name"] = model_name
        #     collection_metadata["model_type"] = model_type

        # if isinstance(relevance_score_fn, str):
        #     assert relevance_score_fn in get_args(PredefinedRelevanceScoreFn)
        #     collection_metadata["hnsw:space"] = relevance_score_fn
        # else:
        #     kwargs["relevance_score_fn"] = relevance_score_fn
        # kwargs["collection_metadata"] = collection_metadata
        # return Chroma(**kwargs)
        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))            
        vectorstore = Chroma(client=persistent_client,
                                collection_name=index_name,
                                embedding_function=query_model) 
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')   
    elif index_type == "RAGatouille":
        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        if init_ragatouille:    
            # Used if the index is not already set, initializes root folder and embedding model
            vectorstore = query_model
        else:   
            # Used if the index is already set, loads the index directly
            vectorstore = RAGPretrainedModel.from_index(index_path=os.path.join(local_db_path,
                                                                                '.ragatouille/colbert/indexes',
                                                                                index_name))
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    else:
        raise NotImplementedError

    if show_progress:
        my_bar.empty()
    return vectorstore

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1,max=60))
def upsert_docs_pinecone(index_name: str,
                         vectorstore: any, 
                         chunker: dict, 
                         batch_size: int = 50, 
                         local_db_path: str = '.'):
    """
    Upserts documents into Pinecone index. Refactored spearately from upsert_docs to allow for tenacity retries.

    Args:
        index_name (str): The name of the Pinecone index.
        vectorstore (any): The vectorstore object for storing the document vectors.
        chunker (dict): The chunker object containing the documents to upsert.
        batch_size (int, optional): The number of documents to upsert in each batch. Defaults to 50.
        local_db_path (str, optional): The path to the local database. Defaults to '.'.

    Returns:
        tuple: A tuple containing the updated vectorstore and retriever objects.
    """
    
    if chunker['rag'] == 'Standard':
        for i in range(0, len(chunker['chunks']), batch_size):
            chunk_batch = chunker['chunks'][i:i + batch_size]
            chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
            vectorstore.add_documents(documents=chunk_batch,
                                        ids=chunk_batch_ids)
        retriever = vectorstore.as_retriever()
    elif chunker['rag'] == 'Parent-Child':
        lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
        store = LocalFileStore(lfs_path)
        
        id_key = "doc_id"
        retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

        for i in range(0, len(chunker['chunks']), batch_size):
            chunk_batch = chunker['chunks'][i:i + batch_size]
            chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
            retriever.vectorstore.add_documents(documents=chunk_batch,
                                                ids=chunk_batch_ids)
        # Index parent docs all at once
        retriever.docstore.mset(list(zip(chunker['pages']['doc_ids'], chunker['pages']['parent_chunks'])))
    elif chunker['rag'] == 'Summary':
        lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
        store = LocalFileStore(lfs_path)
        
        id_key = "doc_id"
        retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

        for i in range(0, len(chunker['summaries']), batch_size):
            chunk_batch = chunker['summaries'][i:i + batch_size]
            chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
            retriever.vectorstore.add_documents(documents=chunk_batch,
                                                ids=chunk_batch_ids)
        # Index parent docs all at once
        retriever.docstore.mset(list(zip(chunker['pages']['doc_ids'], chunker['pages']['docs'])))
    else:
        raise NotImplementedError
    return vectorstore, retriever
        
def upsert_docs(index_type: str, 
                index_name: str,
                vectorstore: any, 
                chunker: dict, 
                batch_size: int = 50, 
                show_progress: bool = False,
                local_db_path: str = '.'):
    """
    Upserts documents into the specified index.

    Args:
        index_type (str): The type of index to upsert the documents into.
        index_name (str): The name of the index.
        vectorstore (any): The vectorstore object to add documents to.
        chunker (dict): The chunker dictionary containing the documents to upsert.
        batch_size (int, optional): The batch size for upserting documents. Defaults to 50.
        show_progress (bool, optional): Whether to show progress during the upsert process. Defaults to False.
        local_db_path (str, optional): The local path to the database folder. Defaults to '.'.

    Returns:
        tuple: A tuple containing the updated vectorstore and retriever objects.
    """
    if show_progress:
        progress_text = "Upsert in progress..."
        my_bar = st.progress(0, text=progress_text)

    if chunker['rag'] == 'Standard':
        # Upsert each chunk in batches
        if index_type == "Pinecone":
            if show_progress:
                progress_text = "Upsert in progress to Pinecone..."
                my_bar.progress(0, text=progress_text)
            vectorstore, retriever=upsert_docs_pinecone(index_name,
                                                        vectorstore, 
                                                        chunker, 
                                                        batch_size, 
                                                        local_db_path)
        elif index_type == "ChromaDB":
            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
                vectorstore.add_documents(documents=chunk_batch,
                                          ids=chunk_batch_ids)
                if show_progress:
                    progress_percentage = i / len(chunker['chunks'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            retriever = vectorstore.as_retriever()
        elif index_type == "RAGatouille":
            # Create an index from the vectorstore.
            # This will default to split documents into 256 tokens each.
            vectorstore.index(
                collection=[chunk.page_content for chunk in chunker['chunks']],
                document_ids=[_stable_hash_meta(chunk.metadata) for chunk in chunker['chunks']],
                index_name=index_name,
                overwrite_index=True,
                split_documents=True,
                document_metadatas=[chunk.metadata for chunk in chunker['chunks']]
            )

            retriever = vectorstore.as_langchain_retriever()
        else:
            raise NotImplementedError
    elif chunker['rag'] == 'Parent-Child':
        if index_type == 'Pincone':
            if show_progress:
                progress_text = "Upsert in progress to Pinecone..."
                my_bar.progress(0, text=progress_text)
            vectorstore, retriever=upsert_docs_pinecone(index_name,
                                                        vectorstore, 
                                                        chunker, 
                                                        batch_size, 
                                                        show_progress,
                                                        local_db_path)
        elif index_type == 'ChromaDB':
            lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
            store = LocalFileStore(lfs_path)
            
            id_key = "doc_id"
            retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
                retriever.vectorstore.add_documents(documents=chunk_batch,
                                                    ids=chunk_batch_ids)
                if show_progress:
                    progress_percentage = i / len(chunker['chunks'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            
            # Index parent docs all at once
            retriever.docstore.mset(list(zip(chunker['pages']['doc_ids'], chunker['pages']['parent_chunks'])))
        elif index_type == "RAGatouille":
            raise Exception('RAGAtouille only supports standard RAG.')
        else:
            raise NotImplementedError
    elif chunker['rag'] == 'Summary':
        if index_type == 'Pincone':
            if show_progress:
                progress_text = "Upsert in progress to Pinecone..."
                my_bar.progress(0, text=progress_text)
            vectorstore, retriever=upsert_docs_pinecone(index_name,
                                                        vectorstore, 
                                                        chunker, 
                                                        batch_size, 
                                                        show_progress,
                                                        local_db_path)
        elif index_type == 'ChromaDB':
            lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
            store = LocalFileStore(lfs_path)
            
            id_key = "doc_id"
            retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

            for i in range(0, len(chunker['summaries']), batch_size):
                chunk_batch = chunker['summaries'][i:i + batch_size]
                chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
                retriever.vectorstore.add_documents(documents=chunk_batch,
                                                    ids=chunk_batch_ids)
                if show_progress:
                    progress_percentage = i / len(chunker['summaries'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            
            # Index parent docs all at once
            retriever.docstore.mset(list(zip(chunker['pages']['doc_ids'], chunker['pages']['docs'])))
        elif index_type == "RAGatouille":
            raise Exception('RAGAtouille only supports standard RAG.')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if show_progress:
        my_bar.empty()
    return vectorstore, retriever

def delete_index(index_type: str, 
                 index_name: str, 
                 rag_type: str,
                 local_db_path: str = '.'):
    """
    Deletes an index based on the specified index type.

    Args:
        index_type (str): The type of index to delete. Valid values are "Pinecone", "ChromaDB", or "RAGatouille".
        index_name (str): The name of the index to delete.
        rag_type (str): The type of RAG (RAGatouille) to delete. Valid values are "Parent-Child" or "Summary".
        local_db_path (str, optional): The path to the local database. Defaults to '.'.

    Raises:
        NotImplementedError: If the index_type is not supported.

    """
    if index_type == "Pinecone":
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        try:
            pc.describe_index(index_name)
            pc.delete_index(index_name)
        except Exception as e:
            # print(f"Error occurred while deleting Pinecone index: {e}")
            pass
        if rag_type == 'Parent-Child' or rag_type == 'Summary':
            try:
                shutil.rmtree(Path(local_db_path).resolve() / 'local_file_store' / index_name)
            except Exception as e:
                # print(f"Error occurred while deleting ChromaDB local_file_store collection: {e}")
                pass    # No need to do anything if it doesn't exist
    elif index_type == "ChromaDB":  
        try:
            persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))
            # indices = persistent_client.list_collections()
            # for idx in indices:
            #     if index_name in idx.name:
                    # persistent_client.delete_collection(name=idx.name)
            persistent_client.delete_collection(name=index_name)
        except Exception as e:
            # print(f"Error occurred while deleting ChromaDB collection: {e}")
            pass
        # Delete local file store if they exist
        if rag_type == 'Parent-Child' or rag_type == 'Summary':
            try:
                shutil.rmtree(Path(local_db_path).resolve() / 'local_file_store' / index_name)
            except Exception as e:
                # print(f"Error occurred while deleting ChromaDB local_file_store collection: {e}")
                pass    # No need to do anything if it doesn't exist
    elif index_type == "RAGatouille":
        try:
            ragatouille_path = os.path.join(local_db_path, '.ragatouille/colbert/indexes', index_name)
            shutil.rmtree(ragatouille_path)
        except Exception as e:
            # print(f"Error occurred while deleting RAGatouille index: {e}")
            pass
    else:
        raise NotImplementedError
def _sanitize_raw_page_data(page):
    """
    Sanitizes the raw page data by removing unnecessary information and checking for meaningful content.
    If pages are merged, this must happen before the merging occurs.

    Args:
        page (Page): The raw page data to be sanitized.

    Returns:
        Page or None: The sanitized page data if it contains meaningful content, otherwise None.
    """
    
    # Yank out some things you'll never care about
    page.metadata['source'] = os.path.basename(page.metadata['source'])   # Strip path
    page.metadata['page'] = int(page.metadata['page']) + 1   # Pages are 0 based, update
    page.page_content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.page_content)   # Merge hyphenated words
    page.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page.page_content.strip())  # Fix newlines in the middle of sentences
    page.page_content = re.sub(r"\n\s*\n", "\n\n", page.page_content)   # Remove multiple newlines

    # Test if there is meaningful content
    text = page.page_content
    num_words = len(text.split())
    if len(text) == 0:
        return None
    alphanumeric_pct = sum(c.isalnum() for c in text) / len(text)
    if num_words < 5 or alphanumeric_pct < 0.3:
        return None
    else:
        return page
def _embedding_size(embedding_family:Union[OpenAIEmbeddings,
                                           VoyageAIEmbeddings,
                                           HuggingFaceInferenceAPIEmbeddings]):
    """
    Returns the size of the embedding for a given embedding model.

    Args:
        embedding_family (object): The embedding model to get the size for.

    Returns:
        int: The size of the embedding.

    Raises:
        NotImplementedError: If the embedding model is not supported.
    """
    # https://platform.openai.com/docs/models/embeddings
    if isinstance(embedding_family,OpenAIEmbeddings):
        name=embedding_family.model
        if name=="text-embedding-ada-002":
            return 1536
        elif name=="text-embedding-3-small":
            return 1536
        elif name=="text-embedding-3-large":
            return 3072
        else:
            raise NotImplementedError(f"The embedding model '{name}' is not available in config.json")
    # https://docs.voyageai.com/embeddings/
    elif isinstance(embedding_family,VoyageAIEmbeddings):
        name=embedding_family.model
        if name=="voyage-2":
            return 1024 
        elif name=="voyage-large-2":
            return 1536
        else:
            raise NotImplementedError(f"The embedding model '{name}' is not available in config.json")
    # See model pages for embedding sizes
    elif isinstance(embedding_family,HuggingFaceInferenceAPIEmbeddings):
        name=embedding_family.model_name
        if name=="sentence-transformers/all-MiniLM-L6-v2":
            return 384
        elif name=="mixedbread-ai/mxbai-embed-large-v1":
            return 1024
        else:
            raise NotImplementedError(f"The embedding model '{name}' is not available in config.json")
    else:
        raise NotImplementedError(f"The embedding family '{embedding_family}' is not available in config.json")

def _stable_hash_meta(metadata: dict) -> str:
    """
    Stable hash of metadata from Langchain Document.

    Args:
        metadata (dict): The metadata dictionary to be hashed.

    Returns:
        str: The hexadecimal representation of the hashed metadata.
    """
    return hashlib.sha1(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
def _check_db_name(index_type:str,index_name:object):
    if index_type=="Pinecone":
        if len(index_name)>45:
            raise ValueError(f'The Pinecone index name must be less than 45 characters. Entry: {index_name}')
    elif index_type=="ChromaDB":
        if len(index_name) > 63:
            raise ValueError(f'The ChromaDB collection name must be less than 63 characters. Entry: {index_name}')
        if not index_name[0].isalnum() or not index_name[-1].isalnum():
            raise ValueError(f'The ChromaDB collection name must start and end with an alphanumeric character. Entry: {index_name}')
        if not re.match(r'^[a-zA-Z0-9_-]+$', index_name):
            raise ValueError(f'The ChromaDB collection name can only contain alphanumeric characters, underscores, or hyphens. Entry: {index_name}')
        if '..' in index_name:
            raise ValueError(f'The ChromaDB collection name cannot contain two consecutive periods. Entry: {index_name}')

### Stuff to test out spotlight

def get_or_create_spotlight_viewer(df:pd.DataFrame,host:str='0.0.0.0',port:int=9000):
    viewers = spotlight.viewers()
    if viewers:
        for viewer in viewers[:-1]:
            viewer.close()
        existing_viewer=spotlight.viewers()[-1]
        return existing_viewer
    
    new_viewer = spotlight.show(df,
                          wait='auto',
                          dtype={"used_by_questions": spotlight_dtypes.SequenceDType(spotlight_dtypes.str_dtype)},
                          host=host,
                          port=port)

    return new_viewer
def get_docs_questions_df(
        docs_db_directory: Path,
        docs_db_collection: str,
        questions_db_directory: Path,
        questions_db_collection: str,
        query_model:object):
    """
    Retrieves and combines documents and questions dataframes.

    Args:
        docs_db_directory (Path): The directory of the documents database.
        docs_db_collection (str): The name of the documents database collection.
        questions_db_directory (Path): The directory of the questions database.
        questions_db_collection (str): The name of the questions database collection.
        query_model (object): The query model object.

    Returns:
        pd.DataFrame: The combined dataframe containing documents and questions data.
    """
    # TODO there's definitely a way to not have to pass query_model, since it should be possible to pull from the db, try to remove this in future versions

    # Check if there exists a query database
    chroma_collections = [collection.name for collection in admin.show_chroma_collections(format=False)['message']]
    matching_collection = [collection for collection in chroma_collections if collection == questions_db_collection]
    if len(matching_collection) > 1:
        raise Exception('Matching collection not found or multiple matching collections found.')
    try:
        if not matching_collection:
            raise Exception('Query database not found. Please create a query database using the Chatbot page and a selected index.')
    except Exception as e:
        st.warning(f"{e}")
        st.stop()
    st.markdown(f"Query database found: {questions_db_collection}")

    docs_df = get_docs_df(docs_db_directory, docs_db_collection, query_model)
    docs_df["type"] = "doc"
    questions_df = get_questions_df(questions_db_directory, questions_db_collection, query_model)
    questions_df["type"] = "question"

    questions_df["num_sources"] = questions_df["sources"].apply(len)
    questions_df["first_source"] = questions_df["sources"].apply(
        lambda x: next(iter(x), None)
    )

    if len(questions_df):
        docs_df["used_by_questions"] = docs_df["id"].apply(
            lambda doc_id: questions_df[
                questions_df["sources"].apply(lambda sources: doc_id in sources)
            ]["id"].tolist()
        )
    else:
        docs_df["used_by_questions"] = [[] for _ in range(len(docs_df))]
    docs_df["used_by_num_questions"] = docs_df["used_by_questions"].apply(len)
    docs_df["used_by_question_first"] = docs_df["used_by_questions"].apply(
        lambda x: next(iter(x), None)
    )

    df = pd.concat([docs_df, questions_df], ignore_index=True)
    return df
def get_docs_df(local_db_path: Path, index_name: str, query_model: object):
    """
    Retrieves documents from a Chroma database and returns them as a pandas DataFrame.

    Args:
        local_db_path (Path): The local path to the Chroma database.
        index_name (str): The name of the collection in the Chroma database.
        query_model (object): The embedding function used for querying the database.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved documents, along with their metadata and embeddings.
            The DataFrame has the following columns:
            - id: The ID of the document.
            - source: The source of the document.
            - page: The page number of the document (default: -1 if not available).
            - document: The content of the document.
            - embedding: The embedding of the document.
    """
    persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))            
    vectorstore = Chroma(client=persistent_client,
                            collection_name=index_name,
                            embedding_function=query_model) 

    response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
    return pd.DataFrame(
        {
            "id": response["ids"],
            "source": [metadata.get("source") for metadata in response["metadatas"]],
            "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
            "document": response["documents"],
            "embedding": response["embeddings"],
        }
    )
def get_questions_df(local_db_path: Path, index_name: str, query_model: object):
    """
    Retrieves questions and related information from a Chroma database and returns them as a pandas DataFrame.

    Args:
        local_db_path (Path): The local path to the Chroma database.
        index_name (str): The name of the collection in the Chroma database.
        query_model (object): The embedding function used for querying the Chroma database.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved questions, answers, sources, and embeddings.
    """
    persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))            
    vectorstore = Chroma(client=persistent_client,
                            collection_name=index_name,
                            embedding_function=query_model) 

    response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
    return pd.DataFrame(
        {
            "id": response["ids"],
            "question": response["documents"],
            "answer": [metadata.get("answer") for metadata in response["metadatas"]],
            "sources": [
                metadata.get("sources").split(",") for metadata in response["metadatas"]
            ],
            "embedding": response["embeddings"],
        }
    )
def add_clusters(df:pd,n_clusters:int,label_llm:object=None,doc_per_cluster:int=5):
    """
    Add clusters to a DataFrame based on the embeddings of its documents.

    Args:
        df (pd.DataFrame): The DataFrame containing the documents and their embeddings.
        n_clusters (int): The number of clusters to create.
        label_llm (object, optional): The label language model to use for generating cluster labels. Defaults to None.
        doc_per_cluster (int, optional): The number of documents to sample per cluster for label generation. Defaults to 5.

    Returns:
        pd.DataFrame: The DataFrame with the added cluster information.
    """
    matrix = np.vstack(df.embedding.values)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df["Cluster"] = labels

    summary=[]
    if label_llm is not None:
        for i in range(n_clusters):
            print(f"Cluster {i} Theme:")
            chunks =  df[df.Cluster == i].document.sample(doc_per_cluster, random_state=42)
            llm_chain = CLUSTER_LABEL | label_llm
            summary.append(llm_chain.invoke(chunks))
            print(summary[-1].content)
        df["Cluster_Label"] = [summary[i].content for i in df["Cluster"]]
    
    return df
def export_to_hf_dataset(df: pd.DataFrame, dataset_name: str):
    """
    Export a pandas DataFrame to a Hugging Face dataset and push it to the Hugging Face Hub.

    Args:
        df (pd.DataFrame): The pandas DataFrame to be exported.
        dataset_name (str): The name of the dataset to be created in the Hugging Face Hub.

    Returns:
        None
    """
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(dataset_name, token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))