"""Centralized dependency management and caching."""

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

def get_cache_data_decorator():
    """Returns appropriate cache_data decorator based on environment"""
    try:
        import streamlit as st
        return st.cache_data
    except:
        # Return no-op decorator when not in Streamlit
        return lambda *args, **kwargs: (lambda func: func)

# Replace @st.cache_data with dynamic decorator
cache_data = get_cache_data_decorator()

class Dependencies:
    """Centralized dependency management with lazy loading."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Dependencies, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    @cache_resource
    def get_llm_deps():
        """Load LLM dependencies."""
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        return ChatOpenAI, ChatAnthropic
    
    @staticmethod
    @cache_resource
    def get_embedding_deps():
        """Load embedding dependencies."""
        from langchain_openai import OpenAIEmbeddings
        from langchain_voyageai import VoyageAIEmbeddings
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        return OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings
    
    @staticmethod
    @cache_resource
    def get_db_deps():
        """Load database dependencies."""
        from pinecone import Pinecone
        import chromadb
        from pinecone import ServerlessSpec
        return Pinecone, chromadb, ServerlessSpec
    
    @staticmethod
    @cache_resource
    def get_vectorstore_deps():
        """Load vector store related dependencies."""
        from langchain_pinecone import PineconeVectorStore
        from langchain_chroma import Chroma
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from langchain.storage import LocalFileStore
        return PineconeVectorStore, Chroma, MultiVectorRetriever, LocalFileStore
    
    @staticmethod
    @cache_resource
    def get_query_deps():
        """Load query processing related dependencies."""
        from operator import itemgetter
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain.schema import format_document
        return format_document, itemgetter, StrOutputParser, RunnableLambda, RunnablePassthrough
    
    @staticmethod
    @cache_resource
    def get_analysis_deps():
        """Load analysis related dependencies."""
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        from datasets import Dataset
        return pd, np, KMeans, Dataset

    @staticmethod
    @cache_resource
    def get_pdf_deps():
        """Load PDF processing dependencies."""
        import fitz
        import requests
        return fitz, requests