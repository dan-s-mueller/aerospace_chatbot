"""Database service implementations."""

from pathlib import Path
import os
from ..core.cache import Dependencies
from ..core.config import ConfigurationError
from ..services.prompts import CLUSTER_LABEL

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
