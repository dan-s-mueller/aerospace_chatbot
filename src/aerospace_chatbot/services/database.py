"""Database service implementations."""

from pathlib import Path
import os
from ..core.cache import Dependencies, get_cache_decorator
from ..core.config import ConfigurationError
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
        
    def initialize_database(self, index_name, embedding_service, rag_type='Standard', namespace=None):
        """Initialize and store database connection."""
        self.index_name = index_name
        self.embedding_service = embedding_service
        self.rag_type = rag_type
        self.namespace = namespace

        # Initialize the vectorstore based on database type
        if self.db_type == 'Pinecone':
            self.vectorstore = self._init_pinecone()
        elif self.db_type == 'ChromaDB':
            self.vectorstore = self._init_chromadb()
        elif self.db_type == 'RAGatouille':
            self.vectorstore = self._init_ragatouille()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        return self.vectorstore

    def delete_index(self, index_name, rag_type):
        """Delete an index from the database."""
        if self.db_type == 'chromadb':
            _, chromadb, _ = self._deps.get_db_deps()
            client = chromadb.PersistentClient(path=str(self.local_db_path / 'chromadb'))
            client.delete_collection(index_name)
            
        elif self.db_type == 'Pinecone':
            Pinecone, _, _ = self._deps.get_db_deps()
            pc = Pinecone()
            pc.delete_index(index_name)
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
                    llm_service,
                    docs_per_cluster: int = 10):
        """Add cluster labels to DataFrame using KMeans clustering."""
        deps = Dependencies()
        pd, np, KMeans, _ = deps.get_analysis_deps()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(np.stack(df['embedding'].values))
        
        # Generate cluster labels using LLM
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
            token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
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
    def get_pinecone_status(self):
        """Get status of Pinecone indexes."""
        api_key = os.getenv('PINECONE_API_KEY')
        return self._cached_pinecone_status(api_key)

    def get_chroma_status(self):
        """Get status of ChromaDB collections."""
        _, chromadb, _ = self._deps.get_db_deps()
        
        if not self.local_db_path:
            return {'status': False, 'message': 'Local database path not set'}
        
        try:
            db_path = self.local_db_path / 'chromadb'
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            if len(collections) == 0:
                return {'status': False, 'message': 'No collections found'}
            return {'status': True, 'collections': collections}
        except Exception as e:
            return {'status': False, 'message': f'Error connecting to ChromaDB: {str(e)}'}

    def get_ragatouille_status(self):
        """Get status of RAGatouille indexes."""
        if not self.local_db_path:
            return {'status': False, 'message': 'Local database path not set'}
        
        try:
            index_path = self.local_db_path / '.ragatouille/colbert/indexes'
            if not index_path.exists():
                return {'status': False, 'message': 'No indexes found'}
            
            indexes = []
            for item in index_path.iterdir():
                if item.is_dir():
                    indexes.append(item.name)
                    
            if len(indexes) == 0:
                return {'status': False, 'message': 'No indexes found'}
            return {'status': True, 'indexes': indexes}
        except Exception as e:
            return {'status': False, 'message': f'Error accessing RAGatouille indexes: {str(e)}'}
    def _init_chromadb(self):
        """Initialize ChromaDB."""
        _, chromadb, _ = self._deps.get_db_deps()
        _, Chroma, _, _ = self._deps.get_vectorstore_deps()
        
        db_path = self.local_db_path / 'chromadb'
        db_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(db_path))
        
        return Chroma(
            client=client,
            collection_name=self.index_name,
            embedding_function=self.embedding_service.get_embeddings()
        )
    def _init_pinecone(self):
        """Initialize Pinecone."""
        Pinecone, _, ServerlessSpec = self._deps.get_db_deps()
        PineconeVectorStore, _, _, _ = self._deps.get_vectorstore_deps()
        
        pc = Pinecone()
        dimension = self.embedding_service.get_dimension()
        
        # Create index if it doesn't exist
        if self.index_name not in [idx.name for idx in pc.list_indexes()]:
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
            
        return PineconeVectorStore(
            index=pc.Index(self.index_name),
            embedding=self.embedding_service.get_embeddings(),
            namespace=self.namespace
        )

    def _init_ragatouille(self):
        """Initialize RAGatouille."""
        from ragatouille import RAGPretrainedModel
        
        index_path = self.local_db_path / '.ragatouille'
        return RAGPretrainedModel.from_pretrained(
            model_name=self.index_name,
            index_root=str(index_path)
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