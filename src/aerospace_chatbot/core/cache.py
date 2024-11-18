import threading

def get_cache_decorator():
    """Returns appropriate cache decorator based on environment"""
    try:
        import streamlit as st
        # Check if we're actually running in a Streamlit environment
        if hasattr(st, '_is_running_with_streamlit'):
            return st.cache_resource
        raise RuntimeError("Not in Streamlit environment")
    except (ImportError, RuntimeError):
        return lambda func: func    # Do nothing

def get_cache_data_decorator():
    """Returns appropriate cache_data decorator based on environment"""
    try:
        import streamlit as st
        # Check if we're actually running in a Streamlit environment
        if hasattr(st, '_is_running_with_streamlit'):
            return st.cache_data
        raise RuntimeError("Not in Streamlit environment")
    except (ImportError, RuntimeError):
        return lambda func: func    # Do nothing


cache_resource = get_cache_decorator()  # Replace @st.cache_resource with dynamic decorator
cache_data = get_cache_data_decorator() # Replace @st.cache_data with dynamic decorator

class Dependencies:
    """Centralized dependency management with lazy loading."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:  # First check without lock
            with cls._lock:        # Only lock if instance might need to be created
                if cls._instance is None:  # Double-check after acquiring lock
                    cls._instance = super(Dependencies, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    class LLM:
        @staticmethod
        @cache_resource
        def get_models():
            """Load LLM model dependencies.
            
            Returns:
                tuple: ChatOpenAI, ChatAnthropic
            """
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            return ChatOpenAI, ChatAnthropic
            
        @staticmethod
        @cache_resource
        def get_chain_utils():
            """Load LLM chain utilities.
            
            Returns:
                tuple: itemgetter, StrOutputParser, RunnableLambda, RunnablePassthrough, ConversationBufferMemory, get_buffer_string, Document, format_document
            """
            from operator import itemgetter
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnableLambda, RunnablePassthrough
            from langchain.memory import ConversationBufferMemory
            from langchain_core.messages import get_buffer_string
            from langchain_core.documents import Document
            from langchain.schema import format_document
            return itemgetter, StrOutputParser, RunnableLambda, RunnablePassthrough, ConversationBufferMemory, get_buffer_string, Document, format_document

    class Embeddings:
        @staticmethod
        @cache_resource
        def get_models():
            """Load embedding model dependencies.
            
            Returns:
                tuple: OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings
            """
            from langchain_openai import OpenAIEmbeddings
            from langchain_voyageai import VoyageAIEmbeddings
            from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            return OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings

    class Storage:
        @staticmethod
        @cache_resource
        def get_vector_stores():
            """Load vector store dependencies.
            
            Returns:
                tuple: PineconeVectorStore, Chroma, MultiVectorRetriever
            """
            from langchain_pinecone import PineconeVectorStore
            from langchain_chroma import Chroma
            from langchain.retrievers.multi_vector import MultiVectorRetriever
            from langchain.storage import LocalFileStore
            return PineconeVectorStore, Chroma, MultiVectorRetriever, LocalFileStore

        @staticmethod
        @cache_resource
        def get_db_clients():
            """Load database client dependencies.
            
            Returns:
                tuple: pinecone_client, chromadb, ServerlessSpec, PersistentClient, RAGPretrainedModel
            """
            from pinecone import Pinecone as pinecone_client
            import chromadb
            from pinecone import ServerlessSpec
            from chromadb import PersistentClient
            from ragatouille import RAGPretrainedModel
            return pinecone_client, chromadb, ServerlessSpec, PersistentClient, RAGPretrainedModel
    class Analysis:
        @staticmethod
        @cache_resource
        def get_tools():
            """Load analysis and data processing dependencies.
            
            Returns:
                tuple: pandas, numpy, KMeans, Dataset
            """
            import pandas as pd
            import numpy as np
            from sklearn.cluster import KMeans
            from datasets import Dataset
            return pd, np, KMeans, Dataset

    class Document:
        @staticmethod
        @cache_resource
        def get_processors():
            """Load document processing dependencies.
            
            Returns:
                tuple: fitz, requests, storage, PyPDFLoader
            """
            import fitz
            import requests
            from google.cloud import storage
            from langchain_community.document_loaders import PyPDFLoader
            return fitz, requests, storage, PyPDFLoader

        @staticmethod
        @cache_resource
        def get_splitters():
            """Load document splitting dependencies.
            
            Returns:
                RecursiveCharacterTextSplitter
            """
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter