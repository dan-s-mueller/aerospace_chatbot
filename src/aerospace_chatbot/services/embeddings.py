"""Embedding service implementations."""

from typing import Any, Dict
from ..core.cache import Dependencies

class EmbeddingService:
    """Manages embedding model operations."""
    
    def __init__(self, model_name, model_type, api_key):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self._embeddings = None
        self._deps = Dependencies()
        
    def get_embeddings(self):
        """Get or create embedding model instance."""
        if self._embeddings is None:
            OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = self._deps.get_embedding_deps()
            
            if self.model_type == 'openai':
                self._embeddings = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=self.api_key
                )
            elif self.model_type == 'voyage':
                self._embeddings = VoyageAIEmbeddings(
                    model=self.model_name,
                    voyage_api_key=self.api_key,
                    truncation=False
                )
            elif self.model_type == 'huggingface':
                self._embeddings = HuggingFaceInferenceAPIEmbeddings(
                    model_name=self.model_name,
                    api_key=self.api_key
                )
                
        return self._embeddings
    
    def get_dimension(self):
        """Get embedding dimension for the selected model."""
        dimensions = {
            'openai': {
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072,
            },
            'voyage': {
                'voyage-large-2': 1536,
                'voyage-large-2-instruct': 1024
            },
            'huggingface': {
                'sentence-transformers/all-MiniLM-L6-v2': 384,
                'mixedbread-ai/mxbai-embed-large-v1': 1024
            }
        }
        
        try:
            return dimensions[self.model_type][self.model_name]
        except KeyError:
            raise NotImplementedError(
                f"The embedding model '{self.model_name}' for type '{self.model_type}' is not available in config"
            )
