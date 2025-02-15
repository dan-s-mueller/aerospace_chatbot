"""Embedding service implementations."""

import os
# from ..core.cache import Dependencies, cache_resource

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import cohere

# Utilities
from langchain_core.documents import Document

# Typing
from typing_extensions import List
from typing import List, Tuple

class EmbeddingService:
    """Manages embedding model operations."""
    
    def __init__(self, model_service, model):
        self.model_service = model_service
        self.model = model
        self._embeddings = None
        
    def get_embeddings(self):
        """Get or create embedding model instance."""
        if self._embeddings is None:
            if self.model_service == 'OpenAI':
                self._embeddings = OpenAIEmbeddings(
                    model=self.model,
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
            elif self.model_service == 'Voyage':
                self._embeddings = VoyageAIEmbeddings(
                    model=self.model,
                    voyage_api_key=os.getenv('VOYAGE_API_KEY'),
                    truncation=False
                )
            elif self.model_service == 'Hugging Face':
                self._embeddings = HuggingFaceInferenceAPIEmbeddings(
                    model_name=self.model,
                    api_key=os.getenv('HUGGINGFACEHUB_API_KEY')
                )
            else:
                raise NotImplementedError(f"The embedding model '{self.model}' for type '{self.model_service}' is not available in config")
            
        return self._embeddings
    
    def get_dimension(self):
        """Get embedding dimension for the selected model."""
        dimensions = {
            'OpenAI': {
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072,
            },
            'Voyage': {
                'voyage-3': 1024,
                'voyage-3-lite': 512
            },
            'Hugging Face': {
                'sentence-transformers/all-MiniLM-L6-v2': 384,
                'mixedbread-ai/mxbai-embed-large-v1': 1024
            },
            'RAGatouille': {
                'colbert-ir/colbertv2.0': 0
            }
        }
        
        try:
            return dimensions[self.model_service][self.model]
        except KeyError:
            raise NotImplementedError(
                f"The embedding model '{self.model}' for type '{self.model_service}' is not available in config"
            )

class RerankService:
    """Manages reranking model operations."""
    
    def __init__(self, model_service, model):
        self.model_service = model_service
        self.model = model
        self._rerank = None
        
    def get_rerank(self):
        """Get or create reranking model instance."""
        if self._rerank is None:
            if self.model_service == 'Cohere':
                self._rerank = cohere.ClientV2(os.getenv('COHERE_API_KEY'))
            else:
                raise NotImplementedError(f"The reranking model '{self.model}' for type '{self.model_service}' is not available in config")
        return self._rerank
    
    def rerank_docs(self, query: str, retrieved_docs: List[Tuple[Document, float]], top_n: int = None):
        """Rerank the retrieved documents."""
        # TODO will default to output cohere formatting. If other models are used, consider updating the class and the rerank method in database.py

        # Cohere's rerank expects a list of strings; we'll supply the page_content, since Langchain Document is the default retrieval type right now.
        inputs = [doc.page_content for doc, _ in retrieved_docs]

        reranker = self.get_rerank()
        response = reranker.rerank(
                model=self.model,
                query=query,
                documents=inputs,
                top_n=top_n,
                return_documents=True
        )
        return response.results