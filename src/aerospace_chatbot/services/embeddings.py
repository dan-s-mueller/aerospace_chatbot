"""Embedding service implementations."""

import os
from ..core.cache import Dependencies

# TODO add cohere embeddings
# TODO add rerank https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311

class EmbeddingService:
    """Manages embedding model operations."""
    
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self._embeddings = None
        self._deps = Dependencies()
        
    def get_embeddings(self):
        """Get or create embedding model instance."""
        if self._embeddings is None:
            OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = self._deps.get_embedding_deps()
            
            if self.model_type == 'OpenAI':
                self._embeddings = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
            elif self.model_type == 'Voyage':
                self._embeddings = VoyageAIEmbeddings(
                    model=self.model_name,
                    voyage_api_key=os.getenv('VOYAGE_API_KEY'),
                    truncation=False
                )
            elif self.model_type == 'Hugging Face':
                self._embeddings = HuggingFaceInferenceAPIEmbeddings(
                    model_name=self.model_name,
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
                'voyage-large-2': 1536,
                'voyage-large-2-instruct': 1024
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
            return dimensions[self.model_type][self.model_name]
        except KeyError:
            raise NotImplementedError(
                f"The embedding model '{self.model_name}' for type '{self.model_type}' is not available in config"
            )
