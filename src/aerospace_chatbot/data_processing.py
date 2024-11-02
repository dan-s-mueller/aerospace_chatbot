import os, re, shutil
import hashlib
from pathlib import Path
from typing import List, Union
import json, jsonlines
import pickle
from tenacity import retry, stop_after_attempt, wait_exponential
from google.cloud import storage
from prompts import CLUSTER_LABEL, SUMMARIZE_TEXT

def get_cache_decorator():
    """Returns appropriate cache decorator based on environment"""
    try:
        import streamlit as st
        return st.cache_resource
    except:
        # Return no-op decorator when not in Streamlit
        return lambda *args, **kwargs: (lambda func: func)

# Replace @st.cache_resource with dynamic decorator
cache_resource = get_cache_decorator()

class DependencyCache:
    """A class to cache dependencies."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    @cache_resource
    def get_db_deps():
        """Load database dependencies only when needed"""
        from pinecone import Pinecone as pinecone_client, ServerlessSpec
        import chromadb
        from chromadb import PersistentClient
        return pinecone_client, chromadb, PersistentClient, ServerlessSpec

    @staticmethod
    @cache_resource
    def get_langchain_deps():
        """Load langchain dependencies only when needed"""
        from langchain_pinecone import PineconeVectorStore
        from langchain_chroma import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from langchain.storage import LocalFileStore
        return PineconeVectorStore, Chroma, RecursiveCharacterTextSplitter, MultiVectorRetriever, LocalFileStore

    @staticmethod
    @cache_resource
    def get_embedding_deps():
        """Load embedding dependencies only when needed"""
        from langchain_openai import OpenAIEmbeddings
        from langchain_voyageai import VoyageAIEmbeddings
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        return OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings

    @staticmethod
    @cache_resource
    def get_doc_deps():
        """Load document dependencies only when needed"""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_core.documents import Document
        from langchain_core.output_parsers import StrOutputParser
        return PyPDFLoader, Document, StrOutputParser

    @staticmethod
    @cache_resource
    def get_spotlight():
        """Load Spotlight only when needed"""
        from renumics import spotlight
        return spotlight

    @staticmethod
    @cache_resource
    def get_analysis_deps():
        """Load analysis dependencies only when needed"""
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        from datasets import Dataset
        return pd, np, KMeans, Dataset

    @staticmethod
    @cache_resource
    def get_admin():
        """Load admin module only when needed"""
        import admin
        return admin

def list_bucket_pdfs(bucket_name: str) -> List[str]:
    """Lists all PDF files in a Google Cloud Storage bucket."""
    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs (files) in the bucket
        blobs = bucket.list_blobs()
        
        # Filter for PDF files and create full GCS paths
        pdf_files = [
            f"gs://{bucket_name}/{blob.name}" 
            for blob in blobs 
            if blob.name.lower().endswith('.pdf')
        ]
        
        return pdf_files
    except Exception as e:
        raise Exception(f"Error accessing GCS bucket: {str(e)}")
def list_available_buckets() -> List[str]:
    """Lists all available buckets in the GCS project."""
    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        
        # List all buckets
        buckets = [bucket.name for bucket in storage_client.list_buckets()]
        
        return buckets
    except Exception as e:
        raise Exception(f"Error accessing GCS buckets: {str(e)}")
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
              namespace=None,
              show_progress:bool=False):
    """Loads documents into the specified index."""
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
                                      show_progress=show_progress,
                                      chunker=chunker)
    vectorstore, _ = upsert_docs(index_type,
                                 index_name,
                                 vectorstore,
                                 chunker,
                                 batch_size=batch_size,
                                 show_progress=show_progress,
                                 local_db_path=local_db_path,
                                 namespace=namespace)
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
    """Chunk the given list of documents into smaller chunks."""
    deps = DependencyCache.get_instance()
    _, _, RecursiveCharacterTextSplitter, _, _ = deps.get_langchain_deps()
    _, SUMMARIZE_TEXT = deps.get_prompts()
    PyPDFLoader, Document, StrOutputParser = deps.get_doc_deps()
    
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
        
        # Load the document. First try as if it is a path, second try as if it is a file object 
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
        return {'rag_type':'Standard',
                'pages':pages,
                'chunks':chunks, 
                'splitters':text_splitter,
                'n_merge_pages':n_merge_pages,
                'chunk_method':chunk_method,
                'chunk_size':chunk_size,
                'chunk_overlap':chunk_overlap}
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
        return {'rag_type':'Parent-Child',
                'pages':{'doc_ids':doc_ids,'parent_chunks':parent_chunks},
                'chunks':chunks,
                'splitters':{'parent_splitter':parent_splitter,'child_splitter':child_splitter},
                'n_merge_pages':n_merge_pages,
                'chunk_method':chunk_method,
                'chunk_size':chunk_size,
                'chunk_overlap':chunk_overlap}
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
        return {'rag_type':'Summary',
                'pages':{'doc_ids':doc_ids,'docs':pages},
                'summaries':summary_docs,
                'llm':llm,
                'n_merge_pages':n_merge_pages,
                'chunk_method':chunk_method,
                'chunk_size':chunk_size,
                'chunk_overlap':chunk_overlap}
    else:
        raise NotImplementedError
@cache_resource
def initialize_database(index_type: str, 
                        index_name: str, 
                        query_model: object,
                        rag_type: str,
                        local_db_path: str = None, 
                        clear: bool = False,
                        init_ragatouille: bool = False,
                        show_progress: bool = False,
                        chunker: dict = None,
                        namespace: str = None):
    """Initialize database with caching for faster subsequent loads"""
    deps = DependencyCache.get_instance()
    pinecone_client, chromadb, PersistentClient, ServerlessSpec = deps.get_db_deps()
    PineconeVectorStore, Chroma, _, _, _ = deps.get_langchain_deps()
    OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = deps.get_embedding_deps()
    
    if show_progress:
        progress_text = "Database initialization..."
        my_bar = st.progress(0, text=progress_text)
    if clear:   # Update metadata if new index  
        # Save chunker metadata
        if chunker is not None:
            # Cannot add objects as metadata, don't add full docs
            index_metadata = {
                key: value for key, value in chunker.items() 
                if key not in ['pages', 'chunks', 'summaries', 'splitters', 'llm'] and value is not None    
            }
        else:
            index_metadata={}
        if isinstance(query_model, OpenAIEmbeddings):
            index_metadata['query_model']= "OpenAI"
            index_metadata['embedding_model'] = query_model.model
        elif isinstance(query_model, VoyageAIEmbeddings):
            index_metadata['query_model'] = "Voyage"
            index_metadata['embedding_model'] = query_model.model
        elif isinstance(query_model, HuggingFaceInferenceAPIEmbeddings):
            index_metadata['query_model'] = "Hugging Face"
            index_metadata['embedding_model'] = query_model.model_name

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
                                        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
                                        namespace=namespace)
        if clear:   # Update metadata if new index
            metadata_vector = [1e-5] * _embedding_size(query_model)  # Empty embedding vector. In queries.py, this is filtered out.
            index.upsert(vectors=[{
                'id': 'db_metadata',
                'values': metadata_vector,
                'metadata': index_metadata
            }])
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    elif index_type == "ChromaDB":
        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))    

        if clear:   # Update metadata if new index
            vectorstore = Chroma(collection_name=index_name,
                                embedding_function=query_model,
                                collection_metadata=index_metadata,
                                client=persistent_client)
        else:
            vectorstore = Chroma(collection_name=index_name,
                                embedding_function=query_model,
                                client=persistent_client)
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')   
    elif index_type == "RAGatouille":
        if chunker is not None:
            print("Warning: The 'chunker' parameter is ignored. It only works with ChromaDB and Pinecone.")
        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        if init_ragatouille:    
            # Used if the index is not already set, initializes root folder and embedding model
            vectorstore = query_model
        else:   
            # Used if the index is already set, loads the index directly using lazy loading
            vectorstore = _init_ragatouille(index_name, local_db_path)
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
                   chunk_batch,
                   chunk_batch_ids,
                   namespace):
    """
    Upserts a batch of documents into a vector database. The lancghain call is identical between Pinecone and ChromaDB.
    This function handles issues with hosted database upserts or when using hugging face or other endpoint services which are less stable.
    """
    if namespace is None:
        vectorstore.add_documents(documents=chunk_batch,
                                  ids=chunk_batch_ids)
    else:
        vectorstore.add_documents(documents=chunk_batch,
                                  ids=chunk_batch_ids,
                                  namespace=namespace)
    return vectorstore
def upsert_docs(index_type, 
                index_name,
                vectorstore, 
                chunker, 
                batch_size = 50, 
                show_progress = False,
                local_db_path = '.',
                namespace = None):
    """Upserts documents into the specified index."""
    deps = DependencyCache.get_instance()
    _, _, _, MultiVectorRetriever, LocalFileStore = deps.get_langchain_deps()
    
    if show_progress:
        progress_text = "Upsert in progress..."
        my_bar = st.progress(0, text=progress_text)
    if chunker['rag_type'] == 'Standard':
        if index_type == "Pinecone" or index_type == "ChromaDB":
            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                chunk_batch_ids = [_stable_hash_meta(chunk.metadata) for chunk in chunk_batch]   # add ID which is the hash of metadata
                
                if index_type == "Pinecone":
                    vectorstore = upsert_docs_db(vectorstore,
                                                 chunk_batch, chunk_batch_ids,
                                                 namespace=namespace)
                else:
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
    elif chunker['rag_type'] == 'Parent-Child':
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
    elif chunker['rag_type'] == 'Summary':
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
    """Deletes an index based on the specified index type."""
    deps = DependencyCache.get_instance()
    pinecone_client, chromadb, _, _ = deps.get_db_deps()
    
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
def copy_pinecone_vectors(index, source_namespace, target_namespace, batch_size:int=100, show_progress:bool=False):
    """
    Copies vectors from a source Pinecone namespace to a target namespace.
    """
    if show_progress:
        progress_text = "Document merging in progress..."
        my_bar = st.progress(0, text=progress_text)
    
    # Obtain list of ids
    ids=[]
    for id in index.list():
        ids.extend(id)
    # Fetch vectors from IDs in chunks (to not overload the API)
    for i in range(0, len(ids), batch_size):
        # Fetch vectors
        fetch_response=index.fetch(ids[i:i+batch_size], 
                                  namespace=source_namespace)

        # Transform the fetched vectors into the correct format for upsert
        vectors_to_upsert = []
        for id, vector_data in fetch_response['vectors'].items():
            vectors_to_upsert.append({
                'id': id,
                'values': vector_data['values'],
                'metadata': vector_data['metadata']
            })

        # Upsert vectors
        index.upsert(vectors=vectors_to_upsert, 
                    namespace=target_namespace)
        if show_progress:
            progress_percentage = i / len(ids)
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    if show_progress:
        my_bar.empty()
def db_name(index_type:str,
            rag_type:str,
            index_name:str,
            model_name:bool=None,
            n_merge_pages:int=None,
            chunk_size:int=None,
            chunk_overlap:int=None,
            check:bool=True):
    """Generates a modified name based on the given parameters."""
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

    # Add chunk parameters if they exist
    if n_merge_pages is not None:
        index_name = index_name + f"-{n_merge_pages}nm"
    if chunk_size is not None:
        index_name = index_name + f"-{chunk_size}"
    if chunk_overlap is not None:
        index_name = index_name + f"-{chunk_overlap}"

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
def get_or_create_spotlight_viewer(df, port=9000):
    """Get or create a Spotlight viewer for the given DataFrame."""
    deps = DependencyCache.get_instance()
    spotlight = deps.get_spotlight()
    
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
def get_docs_questions_df(index_type:str,
                          docs_db_directory: Path,
                          docs_db_collection: str,
                          questions_db_directory: Path,
                          questions_db_collection: str,
                          query_model:object):
    """Retrieves and combines documents and questions dataframes."""
    deps = DependencyCache.get_instance()
    admin = deps.get_admin()
    pd, _, _, _ = deps.get_analysis_deps()
    
    # Check if there exists a query database
    if index_type=='ChromaDB':
        collections = [collection.name for collection in admin.show_chroma_collections(format=False)['message']]
    elif index_type=='Pinecone':
        collections = [collection for collection in admin.show_pinecone_indexes(format=False)['message']]
    matching_collection = [collection for collection in collections if collection == questions_db_collection]
    if len(matching_collection) > 1:
        raise Exception('Matching collection not found or multiple matching collections found.')
    try:
        if not matching_collection:
            raise Exception('Query database not found. Please create a query database using the Chatbot page and a selected index.')
    except Exception as e:
        st.warning(f"{e}")
        st.stop()
    st.markdown(f"Query database found: {questions_db_collection}")

    docs_df = get_docs_df(index_type,docs_db_directory, docs_db_collection, query_model)
    docs_df["type"] = "doc"
    st.markdown(f"Retrieved docs from: {docs_db_collection}")
    questions_df = get_questions_df(questions_db_directory, questions_db_collection, query_model)
    questions_df["type"] = "question"
    st.markdown(f"Retrieved questions from: {questions_db_collection}")

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
    """Retrieves documents from a database and returns them as a pandas DataFrame."""
    deps = DependencyCache.get_instance()
    pinecone_client, chromadb, _, _ = deps.get_db_deps()
    _, Chroma, _, _, _ = deps.get_langchain_deps()
    pd, _, _, _ = deps.get_analysis_deps()
    
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
    """Retrieves questions and related information from a Chroma database."""
    deps = DependencyCache.get_instance()
    _, chromadb, _, _ = deps.get_db_deps()
    _, Chroma, _, _, _ = deps.get_langchain_deps()
    pd, _, _, _ = deps.get_analysis_deps()
    
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
def add_clusters(df, n_clusters:int, label_llm:object=None, doc_per_cluster:int=5):
    """Add clusters to a DataFrame based on the embeddings of its documents."""
    deps = DependencyCache.get_instance()
    pd, np, KMeans, _ = deps.get_analysis_deps()
    CLUSTER_LABEL, _ = deps.get_prompts()
    
    matrix = np.vstack(df.embedding.values)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df["Cluster"] = labels

    summary = []
    if label_llm is not None:
        for i in range(n_clusters):
            print(f"Cluster {i} Theme:")
            chunks = df[df.Cluster == i].document.sample(doc_per_cluster, random_state=42)
            llm_chain = CLUSTER_LABEL | label_llm
            summary.append(llm_chain.invoke(chunks))
            print(summary[-1].content)
        df["Cluster_Label"] = [summary[i].content for i in df["Cluster"]]
    return df
def export_to_hf_dataset(df, dataset_name):
    """Export a pandas DataFrame to a Hugging Face dataset and push it to the Hugging Face Hub."""
    deps = DependencyCache.get_instance()
    _, _, _, Dataset = deps.get_analysis_deps()

    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(dataset_name, token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
def archive_db(index_type:str,index_name:str,query_model:object,export_pickle:bool=False):
    """Archives a database by exporting it to a pickle file."""
    # TODO add function to unarchive db (create chroma db and associated local filestore from pickle file)

    df_temp=get_docs_df(index_type,os.getenv('LOCAL_DB_PATH'), index_name, query_model)

    if export_pickle:
        # Export pickle to db directory
        with open(os.path.join(os.getenv('LOCAL_DB_PATH'),f"archive_{index_type.lower()}_{index_name}.pickle"), "wb") as f:
            pickle.dump(df_temp, f)
    return df_temp
def _sanitize_raw_page_data(page):
    """
    Sanitizes the raw page data by removing unnecessary information and checking for meaningful content.
    If pages are merged, this must happen before the merging occurs.
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
def _embedding_size(embedding_family):
    """Returns the size of the embedding for a given embedding model."""
    deps = DependencyCache.get_instance()
    OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = deps.get_embedding_deps()
    
    # https://platform.openai.com/docs/models/embeddings
    if isinstance(embedding_family, OpenAIEmbeddings):
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
    elif isinstance(embedding_family, VoyageAIEmbeddings):
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
    elif isinstance(embedding_family, HuggingFaceInferenceAPIEmbeddings):
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
    """Calculates the stable hash of the given metadata dictionary."""
    return hashlib.sha1(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
@cache_resource
def _init_ragatouille(index_name: str, local_db_path: str):
    """Lazy load and initialize RAGatouille"""
    import nltk
    nltk.download('punkt', quiet=True)
    from ragatouille import RAGPretrainedModel
    return RAGPretrainedModel.from_index(
        index_path=os.path.join(local_db_path, '.ragatouille/colbert/indexes', index_name)    )
