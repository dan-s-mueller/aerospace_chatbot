from prompts import SUMMARIZE_TEXT

import os, re, shutil
import hashlib
from pathlib import Path
from typing import List, Union
import pickle

from tenacity import retry, stop_after_attempt, wait_exponential

from pinecone import Pinecone as pinecone_client, ServerlessSpec

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
import time

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
        rag_type (str, optional): The type of RAG (Retrieval-Augmented Generation) to use.
        index_name (str, optional): The name of the index.
        n_merge_pages (int, optional): Number of pages to to merge when loading.
        chunk_method (str, optional): The method to chunk the documents.
        chunk_size (int, optional): The size of each chunk.
        chunk_overlap (int, optional): The overlap between chunks.
        clear (bool, optional): Whether to clear the index before loading new documents.
        file_out (str, optional): The output file path.
        batch_size (int, optional): The batch size for upserting documents.
        local_db_path (str, optional): The local database path.
        llm (optional): The language model to use.
        show_progress (bool, optional): Whether to show progress during the loading process.

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
               chunk_size:int=400,
               chunk_overlap:int=0,
               k_child:int=4,
               llm=None,
               show_progress:bool=False):
    """
    Chunk the given list of documents into smaller chunks based on the specified parameters.

    Args:
        docs (List[str]): The list of document paths to be chunked.
        rag_type (str, optional): The type of chunking method to be used.
        chunk_method (str, optional): The method of chunking to be used. None will take whole PDF pages as documents.
        file_out (str, optional): The output file path to save the chunked documents.
        n_merge_pages (int, optional): Number of pages to to merge when loading.
        chunk_size (int, optional): The size of each chunk in tokens. Defaults to 500. Only used if chunk_method is not None.
        chunk_overlap (int, optional): The overlap between chunks in tokens. Defaults to 0. Only used if chunk_method is not None.
        k_child (int, optional): The number of child chunks to split from parnet chunks for 'Parent-Child' rag_type.
        llm (None, optional): The language model to be used for generating summaries.
        show_progress (bool, optional): Whether to show the progress bar during chunking.

    Returns:
        dict: A dictionary containing the chunking results based on the specified rag_type.
    """
    if show_progress:
        progress_text = 'Reading documents...'
        my_bar = st.progress(0, text=progress_text)
    pages=[]
    chunks=[]
    
    # Parse doc pages
    for i, doc in enumerate(docs):
        # Show and update the progress bar
        if show_progress:
            progress_percentage = i / len(docs)
            my_bar.progress(progress_percentage, text=f'Reading documents...{doc}...{progress_percentage*100:.2f}%')
        
        # Load the document
        loader = PyPDFLoader(doc)
        doc_page_data = loader.load()

        # Clean up page info, update some metadata
        doc_pages=[]
        for doc_page in doc_page_data:
            doc_page=_sanitize_raw_page_data(doc_page)
            if doc_page is not None:
                doc_pages.append(doc_page)

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
            # Settings apply to parent splitter. k_child divides parent into smaller sizes.
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                           chunk_overlap=chunk_overlap,
                                                           add_start_index=True)    # Without add_start_index, will not be a unique id
            parent_chunks = parent_splitter.split_documents(pages)
        elif chunk_method=='None':
            parent_splitter = None
            parent_chunks = pages  # No chunking, take whole pages as documents
            # raise ValueError("You must specify a chunk_method with rag_type=Parent-Child.")
        else:
            raise NotImplementedError
        
        # Assign parent doc ids
        doc_ids = [str(_stable_hash_meta(parent_chunk.metadata)) for parent_chunk in parent_chunks]
        
        # Split up child chunks
        id_key = "doc_id"
        chunks = []
        for i, doc in enumerate(parent_chunks):
            _id = doc_ids[i]

            if chunk_method=='character_recursive':
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size/k_child, 
                                                chunk_overlap=chunk_overlap,
                                                add_start_index=True)
            elif chunk_method=='None':
                i_chunk_size=len(doc.page_content)/k_child
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=i_chunk_size, 
                                                chunk_overlap=0,
                                                add_start_index=True)

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
        batch_size_submit = 10  # Set the batch size
        for i in range(0, len(pages), batch_size_submit):
            batch_pages = pages[i:i+batch_size_submit]  # Get the batch of pages
            summary = chain.batch(batch_pages, config={"max_concurrency": batch_size_submit})
            summaries.extend(summary)
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
        local_db_path (str, optional): The path to the local database.
        clear (bool, optional): Whether to clear the index.
        init_ragatouille (bool, optional): Whether to initialize the RAGatouille model.
        show_progress (bool, optional): Whether to show the progress bar.

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
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        
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
        # TODO add in collection metadata
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
def upsert_docs_db(vectorstore,
                         chunk_batch: List[Document],
                         chunk_batch_ids: List[str]):
    """
    Upserts a batch of documents into a vector database. The lancghain call is identical between Pinecone and ChromaDB.
    This function handles issues with hosted database upserts or when using hugging face or other endpoint services which are less stable.

    Args:
        vectorstore (VectorStore): The VectorStore object representing the Pinecone or ChromaDB.
        chunk_batch (List[Document]): A list of Document objects representing the batch of documents to be upserted.
        chunk_batch_ids (List[str]): A list of strings representing the IDs of the documents in the batch.

    Returns:
        VectorStore: The updated VectorStore object after upserting the documents.
    """
    vectorstore.add_documents(documents=chunk_batch,
                              ids=chunk_batch_ids)
    return vectorstore
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
        batch_size (int, optional): The batch size for upserting documents.
        show_progress (bool, optional): Whether to show progress during the upsert process.
        local_db_path (str, optional): The local path to the database folder.

    Returns:
        tuple: A tuple containing the updated vectorstore and retriever objects.
    """
    if show_progress:
        progress_text = "Upsert in progress..."
        my_bar = st.progress(0, text=progress_text)
    if chunker['rag'] == 'Standard':
        if index_type == "Pinecone" or index_type == "ChromaDB":
            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
                
                vectorstore = upsert_docs_db(vectorstore,
                                             chunk_batch, chunk_batch_ids)
                    
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
        if index_type == "Pinecone" or index_type == "ChromaDB":
            lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
            store = LocalFileStore(lfs_path)
            
            id_key = "doc_id"
            retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
                
                retriever.vectorstore = upsert_docs_db(retriever.vectorstore,
                                                       chunk_batch, chunk_batch_ids)
                
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
        if index_type == "Pinecone" or index_type == "ChromaDB":
            lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
            store = LocalFileStore(lfs_path)
            
            id_key = "doc_id"
            retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

            for i in range(0, len(chunker['summaries']), batch_size):
                chunk_batch = chunker['summaries'][i:i + batch_size]
                chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
                
                retriever.vectorstore = upsert_docs_db(retriever.vectorstore,
                                                       chunk_batch, chunk_batch_ids)

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
        local_db_path (str, optional): The path to the local database.

    Raises:
        NotImplementedError: If the index_type is not supported.

    """
    if index_type == "Pinecone":
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        try:
            pc.describe_index(index_name)
            pc.delete_index(index_name)
        except Exception as e:
            pass
        if rag_type == 'Parent-Child' or rag_type == 'Summary':
            try:
                shutil.rmtree(Path(local_db_path).resolve() / 'local_file_store' / index_name)
            except Exception as e:
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
            pass
        # Delete local file store if they exist
        if rag_type == 'Parent-Child' or rag_type == 'Summary':
            try:
                shutil.rmtree(Path(local_db_path).resolve() / 'local_file_store' / index_name)
            except Exception as e:
                pass    # No need to do anything if it doesn't exist
    elif index_type == "RAGatouille":
        try:
            ragatouille_path = os.path.join(local_db_path, '.ragatouille/colbert/indexes', index_name)
            shutil.rmtree(ragatouille_path)
        except Exception as e:
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
        elif name=="voyage-large-2-instruct":
            return 1024
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
def _stable_hash_meta(metadata: dict):
    """
    Calculates the stable hash of the given metadata dictionary.

    Args:
        metadata (dict): The dictionary containing the metadata.

    Returns:
        str: The stable hash of the metadata.

    """
    return hashlib.sha1(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
def db_name(index_type:str,rag_type:str,index_name:str,model_name:bool=None,check:bool=True):
    """
    Generates a modified name based on the given parameters.

    Args:
        index_type (str): The type of index.
        rag_type (str): The type of RAG.
        index_name (str): The name of the index.
        model_name (bool, optional): The name of the model. Defaults to None.
        check (bool, optional): Whether to check the validity of the name. Defaults to True.

    Returns:
        str: The modified index name.
        
    Raises:
        ValueError: If the Pinecone index name is longer than 45 characters.
        ValueError: If the ChromaDB collection name is longer than 63 characters.
        ValueError: If the ChromaDB collection name does not start and end with an alphanumeric character.
        ValueError: If the ChromaDB collection name contains characters other than alphanumeric, underscores, or hyphens.
        ValueError: If the ChromaDB collection name contains two consecutive periods.
    """
    # Modify name if it's an advanded RAG type
    if rag_type == 'Parent-Child':
        index_name = index_name + "-parent-child"
    elif rag_type == 'Summary':
        model_name_temp=model_name.replace(".", "-").replace("/", "-").lower()
        model_name_temp = model_name_temp.split('/')[-1]    # Just get the model name, not org
        model_name_temp = model_name_temp[:3] + model_name_temp[-3:]    # First and last 3 characters
        
        model_name_temp=model_name_temp+"-"+_stable_hash_meta({"model_name":model_name})[:4]

        index_name = index_name + "-" + model_name_temp + "-summary"
    else:
        index_name = index_name
    
    # Check if the name is valid
    if check:
        if index_type=="Pinecone":
            if len(index_name) > 45:
                raise ValueError(f"The Pinecone index name must be less than 45 characters. Entry: {index_name}")
            else:
                return index_name
        elif index_type=="ChromaDB":
            if len(index_name) > 63:
                raise ValueError(f"The ChromaDB collection name must be less than 63 characters. Entry: {index_name}")
            if not index_name[0].isalnum() or not index_name[-1].isalnum():
                raise ValueError(f"The ChromaDB collection name must start and end with an alphanumeric character. Entry: {index_name}")
            if not re.match(r"^[a-zA-Z0-9_-]+$", index_name):
                raise ValueError(f"The ChromaDB collection name can only contain alphanumeric characters, underscores, or hyphens. Entry: {index_name}")
            if ".." in index_name:
                raise ValueError(f"The ChromaDB collection name cannot contain two consecutive periods. Entry: {index_name}")
            else:
                return index_name
def get_or_create_spotlight_viewer(df:pd.DataFrame,port:int=9000):
    """
    Get or create a Spotlight viewer for the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to display in the viewer.
        port (int, optional): The port number to use for the viewer. Defaults to 9000.

    Returns:
        spotlight.viewer: The existing or newly created Spotlight viewer.
    """
    # TODO if you try to close spotlight and reuse a port, you get an error that the port is unavailable
    # TODO The viewer does not work properly with merged documents, it does not show the source column properly
    viewers = spotlight.viewers()
    if viewers:
        for viewer in viewers[:-1]:
            viewer.close()
        existing_viewer=spotlight.viewers()[-1]
        return existing_viewer
    
    new_viewer = spotlight.show(dataset=df,
                                wait='auto',
                                port=port,
                                no_ssl=True)

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
    # TODO extend this functionality to Pinecone

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

    docs_df = get_docs_df('ChromaDB',docs_db_directory, docs_db_collection, query_model)
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
def get_docs_df(index_type: str, local_db_path: Path, index_name: str, query_model: object):
    """
    Retrieves documents from a database and returns them as a pandas DataFrame.

    Args:
        index_type (str): The type of index to use (e.g., "Pinecone", "ChromaDB").
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
    # Find rag_type
    if index_name.endswith('-parent-child'):
        rag_type = 'Parent-Child'
    elif index_name.endswith('-summary'):
        rag_type = 'Summary'
    else:
        rag_type = 'Standard'

    if index_type=='ChromaDB':
        persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))            
        vectorstore = Chroma(client=persistent_client,
                                collection_name=index_name,
                                embedding_function=query_model) 
        response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
        
        df_out=pd.DataFrame(
            {
                "id": response["ids"],
                "source": [metadata.get("source") for metadata in response["metadatas"]],
                "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
                "metadata": response["metadatas"],
                "document": response["documents"],
                "embedding": response["embeddings"],
            })  

    elif index_type=='Pinecone':
        # Connect to index
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(index_name) 
        # Obtain list of ids
        ids=[]
        for id in index.list():
            ids.extend(id)
        # Fetch vectors from IDs in chunks (to not overload the API)
        docs=[]
        chunk_size=200  # 200 doesn't error our
        for i in range(0, len(ids), chunk_size):
            vector=index.fetch(ids[i:i+chunk_size])['vectors']
            vector_data = []
            for key, value in vector.items():
                vector_data.append(value)
            docs.extend(vector_data)

        df_out=pd.DataFrame(
            {
                "id": ids,
                "source": [data['metadata']['source'] for data in docs],
                "page": [data['metadata']['page'] for data in docs],
                "metadata": [{'page':data['metadata']['page'],'source':data['metadata']['source']} for data in docs],
                "document": [data['metadata']['page_content'] for data in docs],
                "embedding": [data['values'] for data in docs],
            })  
        # raise NotImplementedError('Only ChromaDB is supported for now')

    # Add parent-doc or original-doc
    if rag_type!='Standard':
        json_data_list = []
        for i, row in df_out.iterrows():
            doc_id = row['metadata']['doc_id']
            file_path = os.path.join(os.getenv('LOCAL_DB_PATH'),'local_file_store',index_name,f"{doc_id}")
            with open(file_path, "r") as f:
                json_data = json.load(f)
            json_data=json_data['kwargs']['page_content']
            json_data_list.append(json_data)
        if rag_type=='Parent-Child':
            df_out['parent-doc'] = json_data_list
        elif rag_type=='Summary':
            df_out['original-doc'] = json_data_list

    return df_out
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
def add_clusters(df,n_clusters:int,label_llm:object=None,doc_per_cluster:int=5):
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
def archive_db(index_type:str,index_name:str,query_model:object,export_pickle:bool=False):
    df_temp=get_docs_df(index_type,os.getenv('LOCAL_DB_PATH'), index_name, query_model)

    if export_pickle:
        # Export pickle to db directory
        with open(os.path.join(os.getenv('LOCAL_DB_PATH'),f"archive_{index_type.lower()}_{index_name}.pickle"), "wb") as f:
            pickle.dump(df_temp, f)
    return df_temp

# TODO add function to unarchive db (create chroma db and associated local filestore from pickle file)