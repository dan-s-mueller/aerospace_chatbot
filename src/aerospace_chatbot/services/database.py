"""Database service implementations."""

from pathlib import Path
from typing import Optional, List, Dict, Any

from ..core.cache import Dependencies
from ..core.config import ConfigurationError

class DatabaseService:
    """Handles database operations for different vector stores."""
    
    def __init__(self, db_type, local_db_path):
        self.db_type = db_type.lower()
        self.local_db_path = Path(local_db_path)
        self._db = None
        self._deps = Dependencies()
        
    def initialize_database(self, 
                          index_name,
                          embedding_service,
                          rag_type='Standard',
                          namespace=None,
                          clear=False):
        """Initialize or get vector store."""
        if self.db_type == 'chromadb':
            return self._init_chroma(index_name, embedding_service, clear)
        elif self.db_type == 'pinecone':
            return self._init_pinecone(index_name, embedding_service, namespace)
        elif self.db_type == 'ragatouille':
            return self._init_ragatouille(index_name)
        else:
            raise ConfigurationError(f"Unsupported database type: {self.db_type}")

    def _init_chroma(self, index_name, embedding_service, clear):
        """Initialize ChromaDB."""
        _, chromadb, _ = self._deps.get_db_deps()
        _, Chroma, _, _ = self._deps.get_vectorstore_deps()
        
        db_path = self.local_db_path / 'chromadb'
        db_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(db_path))
        
        if clear:
            try:
                client.delete_collection(index_name)
            except ValueError:
                pass
                
        return Chroma(
            client=client,
            collection_name=index_name,
            embedding_function=embedding_service.get_embeddings()
        )

    def _init_pinecone(self, index_name, embedding_service, namespace=None):
        """Initialize Pinecone."""
        Pinecone, _, ServerlessSpec = self._deps.get_db_deps()
        PineconeVectorStore, _, _, _ = self._deps.get_vectorstore_deps()
        
        pc = Pinecone()
        dimension = embedding_service.get_dimension()
        
        # Create index if it doesn't exist
        if index_name not in [idx.name for idx in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
            
        return PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embedding_service.get_embeddings(),
            namespace=namespace
        )

    def _init_ragatouille(self, index_name):
        """Initialize RAGatouille."""
        from ragatouille import RAGPretrainedModel
        
        index_path = self.local_db_path / '.ragatouille'
        return RAGPretrainedModel.from_pretrained(
            model_name=index_name,
            index_root=str(index_path)
        )

    def delete_index(self, index_name, rag_type):
        """Delete an index from the database."""
        if self.db_type == 'chromadb':
            _, chromadb, _ = self._deps.get_db_deps()
            client = chromadb.PersistentClient(path=str(self.local_db_path / 'chromadb'))
            client.delete_collection(index_name)
            
        elif self.db_type == 'pinecone':
            Pinecone, _, _ = self._deps.get_db_deps()
            pc = Pinecone()
            pc.delete_index(index_name)
            
