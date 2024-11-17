"""Embedding service implementations."""

import os
from ..core.cache import Dependencies, cache_resource

class EmbeddingService:
    """Manages embedding model operations."""
    
    def __init__(self, model_service, model):
        self.model_service = model_service
        self.model = model
        self._embeddings = None
        
    def get_embeddings(self):
        """Get or create embedding model instance."""
        if self._embeddings is None:
            OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = Dependencies.Embeddings.get_models()
            
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