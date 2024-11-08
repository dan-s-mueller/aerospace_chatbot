"""Database service implementations."""

from pathlib import Path
import os, shutil
import hashlib
import json
import re

from ..core.cache import Dependencies, get_cache_decorator
from ..services.prompts import CLUSTER_LABEL

cache_data = get_cache_decorator()

class DatabaseService:
    """Handles database operations for different vector stores."""
    
    def __init__(self, db_type, local_db_path=None):
        self.db_type = db_type
        self.local_db_path = local_db_path
        self.vectorstore = None
        self.index_name = None
        self.embedding_service = None
        self.rag_type = None
        self.namespace = None
        self._deps = Dependencies()
        
    def initialize_database(self, index_name, embedding_service, rag_type='Standard', namespace=None, clear=False):
        """Initialize and store database connection.
        
        Args:
            index_name (str): Name of the index to initialize. validate_index_name is not called automatically, use it before calling this function.            embedding_service (EmbeddingService): Service for generating embeddings
            rag_type (str, optional): Type of RAG implementation. Defaults to 'Standard'
            namespace (str, optional): Namespace for the index. Defaults to None
            clear (bool, optional): Whether to delete existing index. Defaults to False
                If True, will delete the index if it exists.
                If False, will use existing index or create new one if it doesn't exist.
        
        Returns:
            VectorStore: Initialized vector store instance
            
        Raises:
            ValueError: If database type is not supported
        """
        self.index_name = index_name

        self.embedding_service = embedding_service
        self.rag_type = rag_type
        self.namespace = namespace

        # Initialize the vectorstore based on database type
        if self.db_type == 'Pinecone':
            self.vectorstore = self._init_pinecone(clear=clear)
        elif self.db_type == 'ChromaDB':
            self.vectorstore = self._init_chromadb(clear=clear)
        elif self.db_type == 'RAGatouille':
            self.vectorstore = self._init_ragatouille(clear=clear)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        return self.vectorstore
    def delete_index(self, index_name):
        """Delete an index from the database."""
        # FIXME won't delete local filestore data
        if self.db_type == 'chromadb':
            _, chromadb, _ = self._deps.get_db_deps()
            
            try:
                db_path = os.path.join(self.local_db_path, 'chromadb')
                client = chromadb.PersistentClient(path=str(db_path))
                
                collections = client.list_collections()
                if any(c.name == self.index_name for c in collections):
                    client.delete_collection(self.index_name)
            except Exception as e:
                print(f"Warning: Failed to delete index {self.index_name}: {str(e)}")
            
        elif self.db_type == 'Pinecone':
            pinecone_client, _, ServerlessSpec = self._deps.get_db_deps()
   
            pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            try:
                # Only attempt deletion if index exists
                if self.index_name in [idx.name for idx in pc.list_indexes()]:
                    pc.delete_index(self.index_name)
            except Exception as e:
                print(f"Warning: Failed to delete index {self.index_name}: {str(e)}")
        elif self.db_type == 'RAGatouille':
            try:
                ragatouille_path = os.path.join(self.local_db_path, '.ragatouille/colbert/indexes', self.index_name)
                shutil.rmtree(ragatouille_path)
            except Exception as e:
                print(f"Warning: Failed to delete index {self.index_name}: {str(e)}")
    def get_retriever(self, k=4):
        """Get configured retriever for the vectorstore."""
        self.retriever = None
        search_kwargs = self._process_retriever_args(k)

        if not self.vectorstore:
            raise ValueError("Database not initialized. Please ensure database is initialized before getting retriever.")

        if self.rag_type == 'Standard':
            self.retriever =  self._get_standard_retriever(search_kwargs)
        elif self.rag_type in ['Parent-Child', 'Summary']:
            self.retriever = self._get_multivector_retriever(search_kwargs)
        else:
            raise NotImplementedError(f"RAG type {self.rag_type} not supported")
    def get_docs_questions_df(self, 
                             index_name,
                             query_index_name,
                             embedding_service):
        """Get documents and questions from database as a DataFrame."""
        deps = Dependencies()
        pd, np, _, _ = deps.get_analysis_deps()
        
        # Get embeddings dimension
        embedding_dim = embedding_service.get_dimension()
        
        # Initialize empty arrays for data
        texts, embeddings, metadata = [], [], []
        
        # Get documents from main index
        docs = self.get_documents(index_name)
        for doc in docs:
            texts.append(doc.page_content)
            embeddings.append(doc.metadata.get('embedding', np.zeros(embedding_dim)))
            metadata.append(doc.metadata)
        
        # Get questions if they exist
        try:
            questions = self.get_documents(query_index_name)
            for q in questions:
                texts.append(q.page_content)
                embeddings.append(q.metadata.get('embedding', np.zeros(embedding_dim)))
                metadata.append(q.metadata)
        except Exception:
            pass  # No questions found
            
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'embedding': embeddings,
            'metadata': metadata
        })
        
        return df
    def add_clusters(self,
                    df,
                    n_clusters,
                    llm_service=None,
                    docs_per_cluster: int = 10):
        """Add cluster labels to DataFrame using KMeans clustering."""
        deps = Dependencies()
        pd, np, KMeans, _ = deps.get_analysis_deps()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(np.stack(df['embedding'].values))
        
        # Generate cluster labels using LLM if provided
        if llm_service is not None:
            cluster_labels = []
            for i in range(n_clusters):
                cluster_docs = df[df['cluster'] == i]['text'].head(docs_per_cluster).tolist()
                prompt = CLUSTER_LABEL.format(
                    documents="\n\n".join(cluster_docs),
                    n=docs_per_cluster
                )
                response = llm_service.get_llm().invoke(prompt)
                cluster_labels.append(response.content.strip())
            
            # Map cluster numbers to labels
            cluster_map = {i: label for i, label in enumerate(cluster_labels)}
            df['cluster_label'] = df['cluster'].map(cluster_map)
        return df
    def export_to_hf_dataset(self, df, dataset_name):
        """Export DataFrame to Hugging Face dataset."""
        deps = Dependencies()
        _, _, _, Dataset = deps.get_analysis_deps()
        
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset.push_to_hub(
            dataset_name, 
            token=os.getenv('HUGGINGFACEHUB_API_KEY')
        )
    def get_available_indexes(self, index_type, embedding_name, rag_type):
        """Get available indexes based on current settings."""
        name = []
        base_name = embedding_name.replace('/', '-')
        
        if index_type == 'ChromaDB':
            try:
                _, chromadb, _ = self._deps.get_db_deps()
                client = chromadb.PersistentClient(path=str(self.local_db_path / 'chromadb'))
                collections = client.list_collections()
                
                for collection in collections:
                    if not collection.name.startswith(base_name) or collection.name.endswith('-queries'):
                        continue
                        
                    if (rag_type == 'Parent-Child' and collection.name.endswith('-parent-child')) or \
                       (rag_type == 'Summary' and collection.name.endswith('-summary')) or \
                       (rag_type == 'Standard' and not collection.name.endswith(('-parent-child', '-summary'))):
                        name.append(collection.name)
            except:
                return []
                
        elif index_type == 'Pinecone':
            try:
                Pinecone, _, _ = self._deps.get_db_deps()
                pc = Pinecone()
                indexes = pc.list_indexes()
                
                for index in indexes:
                    if not index.name.startswith(base_name) or index.name.endswith('-queries'):
                        continue
                        
                    if (rag_type == 'Parent-Child' and index.name.endswith('-parent-child')) or \
                       (rag_type == 'Summary' and index.name.endswith('-summary')) or \
                       (rag_type == 'Standard' and not index.name.endswith(('-parent-child', '-summary'))):
                        name.append(index.name)
            except:
                return []
                
        elif index_type == 'RAGatouille':
            try:
                index_path = self.local_db_path / '.ragatouille/colbert/indexes'
                if index_path.exists():
                    for item in index_path.iterdir():
                        if item.is_dir() and item.name.startswith(base_name):
                            name.append(item.name)
            except:
                return []
        
        return name
    def get_database_status(self, db_type):
        """Get status of database indexes/collections."""
        status_handlers = {
            'Pinecone': self._get_pinecone_status,
            'ChromaDB': self._get_chromadb_status,
            'RAGatouille': self._get_ragatouille_status
        }
        
        handler = status_handlers.get(db_type)
        if not handler:
            return {'status': False, 'message': f'Unsupported database type: {db_type}'}
        
        return handler()

    def _get_pinecone_status(self):
        """Get status of Pinecone indexes."""
        api_key = os.getenv('PINECONE_API_KEY')
        return self._cached_pinecone_status(api_key)

    def _get_chromadb_status(self):
        """Get status of ChromaDB collections."""
        _, chromadb, _ = self._deps.get_db_deps()
        
        if not self.local_db_path:
            return {'status': False, 'message': 'Local database path not set'}
        
        try:
            db_path = self.local_db_path / 'chromadb'
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            if not collections:
                return {'status': False, 'message': 'No collections found'}
            return {'status': True, 'indexes': collections}
        except Exception as e:
            return {'status': False, 'message': f'Error connecting to ChromaDB: {str(e)}'}

    def _get_ragatouille_status(self):
        """Get status of RAGatouille indexes."""
        if not self.local_db_path:
            return {'status': False, 'message': 'Local database path not set'}
        
        try:
            index_path = self.local_db_path / '.ragatouille/colbert/indexes'
            if not index_path.exists():
                return {'status': False, 'message': 'No indexes found'}
            
            indexes = [item.name for item in index_path.iterdir() if item.is_dir()]
            if not indexes:
                return {'status': False, 'message': 'No indexes found'}
            return {'status': True, 'indexes': indexes}
        except Exception as e:
            return {'status': False, 'message': f'Error accessing RAGatouille indexes: {str(e)}'}
    def _init_chromadb(self, clear=False):
        """Initialize ChromaDB."""
        _, chromadb, _ = self._deps.get_db_deps()
        _, Chroma, _, _ = self._deps.get_vectorstore_deps()
        
        db_path = os.path.join(self.local_db_path, 'chromadb')
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_path))
        
        if clear:
            self.delete_index(self.index_name)
        return Chroma(
            client=client,
            collection_name=self.index_name,
            embedding_function=self.embedding_service.get_embeddings()
        )
    def _init_pinecone(self, clear=False):
        """Initialize Pinecone."""
        # Check if index exists
        pinecone_client, _, ServerlessSpec = self._deps.get_db_deps()
        PineconeVectorStore, _, _, _ = self._deps.get_vectorstore_deps()

        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY')) # Need touse the native client since langchain can't upload a dummy embedding :(
        if clear:
            self.delete_index(self.index_name)     
        try:
            pc.describe_index(self.index_name)
        except:
            pc.create_index(
                self.index_name,
                dimension=self.embedding_service.get_dimension(),
                spec=ServerlessSpec(
                    cloud='aws',    
                    region='us-west-2'
                )
            )
        return PineconeVectorStore(
            index=pc.Index(self.index_name),
            index_name=self.index_name,
            embedding=self.embedding_service.get_embeddings(),
            text_key='page_content',
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            namespace=self.namespace
        )

    def _init_ragatouille(self, clear=False):
        """Initialize RAGatouille."""
        from ragatouille import RAGPretrainedModel
        
        index_path = os.path.join(self.local_db_path, '.ragatouille')
        if clear:
            self.delete_index(self.index_name)
        return RAGPretrainedModel.from_pretrained(
            pretrained_model_name_or_path=self.embedding_service.model_name,
            index_root=index_path
        )
    @staticmethod
    @cache_data
    def _cached_pinecone_status(api_key):
        """Helper function to cache the processed Pinecone status."""
        Pinecone, _, _ = Dependencies().get_db_deps()
        
        if not api_key:
            return {'status': False, 'message': 'Pinecone API Key is not set'}
        
        try:
            pc = Pinecone(api_key=api_key)
            indexes = pc.list_indexes()
            if len(indexes) == 0:
                return {'status': False, 'message': 'No indexes found'}
            # Convert indexes to list of names to avoid caching Pinecone objects
            return {'status': True, 'indexes': [idx.name for idx in indexes]}
        except Exception as e:
            return {'status': False, 'message': f'Error connecting to Pinecone: {str(e)}'}
    def _get_standard_retriever(self, search_kwargs):
        """Get standard retriever based on index type."""
        if self.db_type in ['Pinecone', 'ChromaDB']:
            return self.vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs=search_kwargs
            )
        elif self.db_type == 'RAGatouille':
            return self.vectorstore.as_langchain_retriever(
                k=search_kwargs['k']
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    def _get_multivector_retriever(self, search_kwargs):
        """Get multi-vector retriever for Parent-Child or Summary RAG types."""
        # FIXME check that hashing/indexing with parent/child is working. Doesn't look like it is. See documents.py for chunking hashing.
        LocalFileStore = self._deps.get_core_deps()[2]
        MultiVectorRetriever = self._deps.get_core_deps()[1]
        
        lfs = LocalFileStore(
            Path(self.local_db_path).resolve() / 'local_file_store' / self.index_name
        )
        return MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=lfs,
            id_key="doc_id",
            search_kwargs=search_kwargs
        )

    def _process_retriever_args(self, k=4):
        """Process the retriever arguments."""
        # Set up filter
        if self.db_type == 'Pinecone':
            filter_kwargs = {"type": {"$ne": "db_metadata"}}
        else:
            filter_kwargs = None

        # Implement filtering and number of documents to return
        search_kwargs = {'k': k}
        
        if filter_kwargs:
            search_kwargs['filter'] = filter_kwargs
            
        return search_kwargs
def validate_index_name(base_index_name,
                       db_service,
                       doc_processor):
    """Validate and format index name with optional parameters."""
    from ..processing.documents import DocumentProcessor

    if not base_index_name or not base_index_name.strip():
        raise ValueError("Index name cannot be empty or contain only whitespace")
    
    # Clean the base name
    name = base_index_name.lower().strip()
    
    # Add RAG type suffix
    if doc_processor.rag_type == 'Parent-Child':
        name += '-parent-child'
    elif doc_processor.rag_type == 'Summary':
        if not doc_processor.embedding_service or not doc_processor.embedding_service.model_name:
            raise ValueError("Embedding service with model name is required for Summary RAG type")
        
        model_name = doc_processor.embedding_service.model_name
        model_name_temp = model_name.replace(".", "-").replace("/", "-").lower()
        model_name_temp = model_name_temp.split('/')[-1]  # Just get the model name, not org
        model_name_temp = model_name_temp[:3] + model_name_temp[-3:]  # First and last 3 characters
        
        # Add hash of model name
        model_name_temp += "-" + DocumentProcessor.stable_hash_meta({"model_name": model_name})[:4]
        name += f"-{model_name_temp}-summary"
    
    # Add parameters if they exist
    if doc_processor.merge_pages is not None:
        name += f"-{doc_processor.merge_pages}nm"
    else:
        if doc_processor.chunk_size is not None:
            name += f"-{doc_processor.chunk_size}"
        if doc_processor.chunk_overlap is not None:
            name += f"-{doc_processor.chunk_overlap}"

    # Database-specific validation
    if db_service.db_type == "Pinecone":
        if len(name) > 45:
            raise ValueError(f"The Pinecone index name must be less than 45 characters. Entry: {name}")
    elif db_service.db_type == "ChromaDB":
        if len(name) > 63:
            raise ValueError(f"The ChromaDB collection name must be less than 63 characters. Entry: {name}")
        if not name[0].isalnum() or not name[-1].isalnum():
            raise ValueError(f"The ChromaDB collection name must start and end with an alphanumeric character. Entry: {name}")
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"The ChromaDB collection name can only contain alphanumeric characters, underscores, or hyphens. Entry: {name}")
        if ".." in name:
            raise ValueError(f"The ChromaDB collection name cannot contain two consecutive periods. Entry: {name}")
    
    return name