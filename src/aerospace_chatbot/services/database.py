"""Database service implementations."""

import os, shutil
import re

from ..core.cache import Dependencies, get_cache_decorator
from ..services.prompts import CLUSTER_LABEL

cache_data = get_cache_decorator()

class DatabaseService:
    """Handles database operations for different vector stores."""
    
    def __init__(self, db_type, index_name, rag_type, embedding_service):
        self.db_type = db_type
        self.index_name = index_name
        self.rag_type = rag_type
        self.embedding_service = embedding_service
        self.vectorstore = None
        self.namespace = None
        self._deps = Dependencies()
        
    def initialize_database(self, namespace=None, clear=False):
        """Initialize and store database connection."""
        self.namespace = namespace

        print(f"Embedding Service in initialize_database: {self.embedding_service.get_embeddings()}")

        # Check if LOCAL_DB_PATH environment variable exists
        if not os.getenv('LOCAL_DB_PATH'):
            raise ValueError("LOCAL_DB_PATH environment variable must be set")

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
    def delete_index(self):
        """Delete an index from the database."""
        # Need to decide if a databaseprocessor can have multiple index names, this only processes the name assigned to the processor, assuming one per processor.
        # FIXME won't delete local filestore data
        if self.db_type == 'chromadb':
            _, chromadb, _ = self._deps.get_db_deps()
            
            try:
                db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
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
                ragatouille_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille/colbert/indexes', self.index_name)
                shutil.rmtree(ragatouille_path)
            except Exception as e:
                print(f"Warning: Failed to delete index {self.index_name}: {str(e)}")
    def index_documents(self,
                       chunking_result,
                       batch_size=100,
                       clear=False,
                       show_progress=False):
        """Index processed documents. This is where the index is initialized or created if required."""
        OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = self._deps.get_embedding_deps()
        # TODO update with dependency cache
        import streamlit as st
        from ..processing.documents import DocumentProcessor


        if show_progress:  
            progress_text = "Document indexing in progress..."
            my_bar = st.progress(0, text=progress_text)

        print(f"Embedding Service in index_documents: {self.embedding_service.get_embeddings()}")
        self.vectorstore = self.initialize_database(
            namespace=self.namespace,
            clear=clear
        )

        # Add index metadata
        index_metadata = {}
        if chunking_result.merge_pages is not None:
            index_metadata['merge_pages'] = chunking_result.merge_pages
        if chunking_result.chunk_method is not 'None':
            index_metadata['chunk_method'] = chunking_result.chunk_method
        if chunking_result.chunk_size is not None:
            index_metadata['chunk_size'] = chunking_result.chunk_size
        if chunking_result.chunk_overlap is not None:
            index_metadata['chunk_overlap'] = chunking_result.chunk_overlap
        if isinstance(self.embedding_service.get_embeddings(), OpenAIEmbeddings):
            index_metadata['query_model']= "OpenAI"
            index_metadata['embedding_model'] = self.embedding_service.get_embeddings().model
        elif isinstance(self.embedding_service.get_embeddings(), VoyageAIEmbeddings):
            index_metadata['query_model'] = "Voyage"
            index_metadata['embedding_model'] = self.embedding_service.get_embeddings().model
        elif isinstance(self.embedding_service.get_embeddings(), HuggingFaceInferenceAPIEmbeddings):
            index_metadata['query_model'] = "Hugging Face"
            index_metadata['embedding_model'] = self.embedding_service.get_embeddings().model_name
        self._store_index_metadata(index_metadata)

        # Validate index name and parameters
        self._validate_index(chunking_result)

        # Index chunks in batches
        for i in range(0, len(chunking_result.chunks), batch_size):
            batch = chunking_result.chunks[i:i + batch_size]
            batch_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in batch]
            
            if self.db_type == "Pinecone":
                self.vectorstore.add_documents(documents=batch, ids=batch_ids, namespace=self.namespace)
            else:
                self.vectorstore.add_documents(documents=batch, ids=batch_ids)

            if show_progress:
                progress_percentage = min(1.0, (i + batch_size) / len(chunking_result.chunks))
                my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')

        # Handle parent documents or summaries if needed
        # FIXME this won't work, loop previous also needs updating
        if self.rag_type in ['Parent-Child', 'Summary']:
            self._store_parent_docs(chunking_result)

        if show_progress:
            my_bar.empty()
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
                             query_index_name):
        """Get documents and questions from database as a DataFrame."""
        # FIXME get query_index_name from databaseservice object.
        deps = Dependencies()
        pd, np, _, _ = deps.get_analysis_deps()
        
        # Get embeddings dimension
        embedding_dim = self.embedding_service.get_dimension()
        
        # Initialize empty arrays for data
        texts, embeddings, metadata = [], [], []
        
        # Get documents from main index
        docs = self.get_documents(self.index_name)
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
    def get_available_indexes(self):
        """Get available indexes based on current settings."""
        # FIXME do this filtering based on index metadata, it won't work properly
        name = []
        base_name = self.embedding_service.model_name.replace('/', '-')
        
        if self.db_type == 'ChromaDB':
            try:
                _, chromadb, _ = self._deps.get_db_deps()
                client = chromadb.PersistentClient(path=os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb'))
                collections = client.list_collections()
                
                for collection in collections:
                    if not collection.name.startswith(base_name) or collection.name.endswith('-queries'):
                        continue
                        
                    if (self.rag_type == 'Parent-Child' and collection.name.endswith('-parent-child')) or \
                       (self.rag_type == 'Summary' and collection.name.endswith('-summary')) or \
                       (self.rag_type == 'Standard' and not collection.name.endswith(('-parent-child', '-summary'))):
                        name.append(collection.name)
            except:
                return []
                
        elif self.db_type == 'Pinecone':
            try:
                Pinecone, _, _ = self._deps.get_db_deps()
                pc = Pinecone()
                indexes = pc.list_indexes()
                
                for index in indexes:
                    if not index.name.startswith(base_name) or index.name.endswith('-queries'):
                        continue
                        
                    if (self.rag_type == 'Parent-Child' and index.name.endswith('-parent-child')) or \
                       (self.rag_type == 'Summary' and index.name.endswith('-summary')) or \
                       (self.rag_type == 'Standard' and not index.name.endswith(('-parent-child', '-summary'))):
                        name.append(index.name)
            except:
                return []
                
        elif self.db_type == 'RAGatouille':
            try:
                index_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille/colbert/indexes')
                if os.path.exists(index_path):
                    for item in os.listdir(index_path):
                        if os.path.isdir(os.path.join(index_path, item)) and item.startswith(base_name):
                            name.append(item)
            except:
                return []
        
        return name
    @staticmethod
    def get_database_status(self,db_type):
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

    def copy_vectors(self, source_namespace, target_namespace, batch_size=100, show_progress=False):
        # FIXME, moved over without testing from documents
        """Copies vectors from a source Pinecone namespace to a target namespace."""
        # TODO update with dependency cache
        import streamlit as st
        if self.db_type != 'Pinecone':
            raise ValueError("Vector copying is only supported for Pinecone databases")
            
        # FIXME initialize database first, this function deosn't exist
        index = self.get_index(self.index_name)
        
        if show_progress:
            try:
                progress_text = "Document merging in progress..."
                my_bar = st.progress(0, text=progress_text)
            except ImportError:
                show_progress = False
        
        # Get list of all vector IDs
        ids = []
        for id_batch in index.list():
            ids.extend(id_batch)
            
        # Process vectors in batches
        for i in range(0, len(ids), batch_size):
            # Fetch vectors
            batch_ids = ids[i:i + batch_size]
            fetch_response = index.fetch(batch_ids, namespace=source_namespace)
            
            # Transform fetched vectors for upsert
            vectors_to_upsert = [
                {
                    'id': id,
                    'values': vector_data['values'],
                    'metadata': vector_data['metadata']
                }
                for id, vector_data in fetch_response['vectors'].items()
            ]
            
            # Upsert vectors to target namespace
            index.upsert(vectors=vectors_to_upsert, namespace=target_namespace)
            
            if show_progress:
                progress_percentage = min(1.0, (i + batch_size) / len(ids))
                my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
        
        if show_progress:
            my_bar.empty()
    
    @staticmethod
    def _get_pinecone_status(self):
        """Get status of Pinecone indexes."""
        api_key = os.getenv('PINECONE_API_KEY')
        return self._cached_pinecone_status(api_key)
    
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

    @staticmethod
    def _get_chromadb_status(self):
        """Get status of ChromaDB collections."""
        _, chromadb, _ = self._deps.get_db_deps()
        
        if not os.getenv('LOCAL_DB_PATH'):
            return {'status': False, 'message': 'Local database path not set'}
        
        try:
            db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            if not collections:
                return {'status': False, 'message': 'No collections found'}
            return {'status': True, 'indexes': collections}
        except Exception as e:
            return {'status': False, 'message': f'Error connecting to ChromaDB: {str(e)}'}

    @staticmethod
    def _get_ragatouille_status(self):
        """Get status of RAGatouille indexes."""
        if not os.getenv('LOCAL_DB_PATH'):
            return {'status': False, 'message': 'Local database path not set'}
        
        try:
            index_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille/colbert/indexes')
            if not os.path.exists(index_path):
                return {'status': False, 'message': 'No indexes found'}
            
            indexes = [item for item in os.listdir(index_path) if os.path.isdir(os.path.join(index_path, item))]
            if not indexes:
                return {'status': False, 'message': 'No indexes found'}
            return {'status': True, 'indexes': indexes}
        except Exception as e:
            return {'status': False, 'message': f'Error accessing RAGatouille indexes: {str(e)}'}
        
    def _init_chromadb(self, clear=False):
        """Initialize ChromaDB."""
        _, chromadb, _ = self._deps.get_db_deps()
        _, Chroma, _, _ = self._deps.get_vectorstore_deps()
        
        db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        
        if clear:
            self.delete_index()

        print(f"Embedding Service in _init_chromadb: {self.embedding_service.get_embeddings()}")

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
            self.delete_index()     
        try:
            pc.describe_index()
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
        
        index_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille')
        if clear:
            self.delete_index()
        return RAGPretrainedModel.from_pretrained(
            pretrained_model_name_or_path=self.embedding_service.model_name,
            index_root=index_path
        )
    
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
            os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', self.index_name)
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
    def _store_index_metadata(self, index_metadata):
        """Store index metadata based on the database type."""

        embedding_size = self.embedding_service.get_dimension()
        metadata_vector = [1e-5] * embedding_size

        if self.db_type == "Pinecone":
            # TODO use cache dependency function
            # TODO see if I can do this with langchain vectorstore, problem is I need to make a dunmmy embedding.
            from pinecone import Pinecone as pinecone_client
            
            pc_native = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc_native.Index(self.index_name)
            index.upsert(vectors=[{
                'id': 'db_metadata',
                'values': metadata_vector,
                'metadata': index_metadata
            }])
        
        elif self.db_type == "Chroma":
            # TODO use cache dependency function
            from chromadb import PersistentClient
            chroma_native = PersistentClient(path=os.path.join(os.getenv('LOCAL_DB_PATH'),'chromadb'))    
            index = chroma_native.get_collection(name=self.index_name)
            index.add(
                embeddings=[metadata_vector],
                metadatas=[index_metadata],
                ids=['db_metadata']
            )

        elif self.db_type == "RAGatouille":
            # TODO add metadata storage for RAGatouille, maybe can use the same method as others
            print("Warning: Metadata storage is not yet supported for RAGatouille indexes")
    def _validate_index(self, doc_processor):
        """Validate and format index with parameters passed from DocumentProcessor. Does nothing if no exceptions are raised, unless RAG type is Parent-Child or Summary, which will append to index_name."""
        from ..processing.documents import DocumentProcessor

        if doc_processor.rag_type != self.rag_type:
            raise ValueError(f"RAG type mismatch: DocumentProcessor has '{doc_processor.rag_type}' but DatabaseService has '{self.rag_type}'")
        if not self.index_name or not self.index_name.strip():
            raise ValueError("Index name cannot be empty or contain only whitespace")
        
        # Clean the base name
        name = self.index_name.lower().strip()
        
        # Add RAG type suffix
        if doc_processor.rag_type == 'Parent-Child':
            name += '-parent-child'
            self.index_name = name
        elif doc_processor.rag_type == 'Summary':
            if not doc_processor.llm_service:
                raise ValueError("LLM service is required for Summary RAG type")
            
            summary_model_name = doc_processor.embedding_service.model_name
            model_name_temp = summary_model_name.replace(".", "-").replace("/", "-").lower()
            model_name_temp = model_name_temp.split('/')[-1]  # Just get the model name, not org
            model_name_temp = model_name_temp[:3] + model_name_temp[-3:]  # First and last 3 characters
            
            # Add hash of model name
            name += "-summary"
            self.index_name = name

        # Database-specific validation
        if self.db_type == "Pinecone":
            if len(name) > 45:
                raise ValueError(f"The Pinecone index name must be less than 45 characters. Entry: {name}")
        elif self.db_type == "ChromaDB":
            print(f"ChromaDB name: {name}")
            if len(name) > 63:
                raise ValueError(f"The ChromaDB collection name must be less than 63 characters. Entry: {name}")
            if not name[0].isalnum() or not name[-1].isalnum():
                raise ValueError(f"The ChromaDB collection name must start and end with an alphanumeric character. Entry: {name}")
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                raise ValueError(f"The ChromaDB collection name can only contain alphanumeric characters, underscores, or hyphens. Entry: {name}")
            if ".." in name:
                raise ValueError(f"The ChromaDB collection name cannot contain two consecutive periods. Entry: {name}")