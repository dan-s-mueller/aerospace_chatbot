"""Database service implementations."""

import os, shutil
import re
from pathlib import Path
import time

from ..core.cache import Dependencies
from ..services.prompts import CLUSTER_LABEL
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pinecone.exceptions import NotFoundException, PineconeApiException

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

    def index_documents(self,
                       chunking_result,
                       batch_size=100,
                       clear=False,
                       show_progress=False):
        """Index processed documents. This is where the index is initialized or created if required."""

        # Initialize database
        self.initialize_database(
            namespace=self.namespace,
            clear=clear
        )

        # Add index metadata
        self._store_index_metadata(chunking_result)

        # Validate index name and parameters, upsert documents
        self._validate_index(chunking_result)
        self._upsert_docs(chunking_result, batch_size, show_progress, clear=clear)

    def get_retriever(self, k=4):
        """Get configured retriever for the vectorstore."""
        self.retriever = None
        search_kwargs = self._process_retriever_args(k)

        if not self.vectorstore:
            raise ValueError("Database not initialized. Please ensure database is initialized before getting retriever.")

        if self.rag_type == 'Standard':
            self._get_standard_retriever(search_kwargs)
        elif self.rag_type in ['Parent-Child', 'Summary']:
            self._get_multivector_retriever(search_kwargs)
        else:
            raise NotImplementedError(f"RAG type {self.rag_type} not supported")
    
    def copy_vectors(self, source_namespace, target_namespace, batch_size=100, show_progress=False):
        # FIXME, moved over without testing from documents
        """Copies vectors from a source Pinecone namespace to a target namespace."""
        if self.db_type != 'Pinecone':
            raise ValueError("Vector copying is only supported for Pinecone databases")
            
        # FIXME initialize database first, this function deosn't exist
        index = self.get_index(self.index_name)
        
        # if show_progress:
        #     try:
        #         # TODO update with dependency cache
        #         import streamlit as st
        #         progress_text = "Document merging in progress..."
        #         my_bar = st.progress(0, text=progress_text)
        #     except ImportError:
        #         show_progress = False
        
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
            
            # if show_progress:
            #     progress_percentage = min(1.0, (i + batch_size) / len(ids))
            #     my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
        
        # if show_progress:
        #     my_bar.empty()

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
                
    def get_docs_questions_df(self, query_index_name):
        """Get documents and questions from database as a DataFrame."""
        # FIXME, this requires a refactor from data_processing, it's missing the majority of the functionality
        # FIXME get query_index_name from databaseservice object.
        
        # deps = Dependencies()
        # pd, np, _, _ = deps.get_analysis_deps()
        # admin = deps.get_admin()
        import pandas as pd
        
        # Check if query database exists
        # FIXME, hack utils._display_database_status to get this
        if self.db_type == 'ChromaDB':
            collections = [collection.name for collection in admin.show_chroma_collections(format=False)['message']]
        elif self.db_type == 'Pinecone':
            collections = [collection for collection in admin.show_pinecone_indexes(format=False)['message']]
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        matching_collection = [collection for collection in collections if collection == query_index_name]
        if len(matching_collection) > 1:
            raise Exception('Matching collection not found or multiple matching collections found.')
        try:
            if not matching_collection:
                raise Exception('Query database not found. Please create a query database using the Chatbot page and a selected index.')
        except Exception as e:
            import streamlit as st
            st.warning(f"{e}")
            st.stop()
        
        # import streamlit as st
        # st.markdown(f"Query database found: {query_index_name}")

        # Get documents dataframe
        docs_df = self._get_docs_df()
        docs_df["type"] = "doc"
        # st.markdown(f"Retrieved docs from: {self.index_name}")

        # Get questions dataframe
        questions_df = self._get_questions_df(query_index_name)
        questions_df["type"] = "question"
        # st.markdown(f"Retrieved questions from: {query_index_name}")

        # Process questions metadata
        questions_df["num_sources"] = questions_df["sources"].apply(len)
        questions_df["first_source"] = questions_df["sources"].apply(
            lambda x: next(iter(x), None)
        )

        # Process document usage by questions
        if len(questions_df):
            docs_df["used_by_questions"] = docs_df["id"].apply(
                lambda doc_id: questions_df[
                    questions_df["sources"].apply(lambda sources: doc_id in sources)
                ]["id"].tolist()
            )
        else:
            docs_df["used_by_questions"] = [[] for _ in range(len(docs_df))]
        
        docs_df["used_by_num_questions"] = docs_df["used_by_questions"].apply(len)
        docs_df["used_by_question_first"] = docs_df["used_by_questions"].apply(
            lambda x: next(iter(x), None)
        )

        # Combine dataframes
        df = pd.concat([docs_df, questions_df], ignore_index=True)
        return df

    def _get_docs_df(self):
        """Helper method to get documents dataframe."""
        deps = Dependencies()
        pd, np, _, _ = deps.get_analysis_deps()
        
        # Get embeddings dimension
        embedding_dim = self.embedding_service.get_dimension()
        
        # Get documents from main index
        docs = self.get_documents(self.index_name)
        
        # Create dataframe
        return pd.DataFrame({
            'id': [doc.metadata.get('id', '') for doc in docs],
            'text': [doc.page_content for doc in docs],
            'embedding': [doc.metadata.get('embedding', np.zeros(embedding_dim)) for doc in docs],
            'metadata': [doc.metadata for doc in docs]
        })

    def _get_questions_df(self, query_index_name):
        """Helper method to get questions dataframe."""
        deps = Dependencies()
        pd, np, _, _ = deps.get_analysis_deps()
        
        # Get embeddings dimension  
        embedding_dim = self.embedding_service.get_dimension()
        
        try:
            questions = self.get_documents(query_index_name)
            
            # Create dataframe
            df = pd.DataFrame({
                'id': [q.metadata.get('id', '') for q in questions],
                'text': [q.page_content for q in questions],
                'embedding': [q.metadata.get('embedding', np.zeros(embedding_dim)) for q in questions],
                'metadata': [q.metadata for q in questions],
                'answer': [q.metadata.get('answer', '') for q in questions],
                'sources': [q.metadata.get('sources', '').split(',') for q in questions]
            })
            return df
        except Exception:
            return pd.DataFrame(columns=['id', 'text', 'embedding', 'metadata', 'answer', 'sources'])

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
    @staticmethod
    def get_available_indexes(db_type, embedding_model, rag_type):
        """Get available indexes based on current settings."""
        # FIXME do this filtering based on index metadata, it won't work properly
        available_indexes = []
        index_metadatas = []

        def _check_get_index_criteria(index_name):
            """Check if index meets RAG type criteria."""
            if index_name.endswith('-queries'):
                return False
                
            return (rag_type == 'Parent-Child' and index_name.endswith('-parent-child')) or \
                   (rag_type == 'Summary' and index_name.endswith('-summary')) or \
                   (rag_type == 'Standard' and not index_name.endswith(('-parent-child', '-summary')))

        def _get_chromadb_indexes():
            """Get available ChromaDB indexes."""
            import chromadb
            client = chromadb.PersistentClient(path=os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb'))
            indexes = client.list_collections()
            
            for index in indexes:
                if not _check_get_index_criteria(index.name):
                    continue
                collection = client.get_collection(index.name)
                metadata = collection.get(ids=['db_metadata'])
                if metadata and metadata['metadatas'][0].get('embedding_model') == embedding_model:
                    available_indexes.append(index.name)
                    index_metadatas.append(metadata['metadatas'][0])

        def _get_pinecone_indexes():
            """Get available Pinecone indexes."""
            from pinecone import Pinecone
            pc = Pinecone()
            indexes = pc.list_indexes()
            
            for index in indexes:
                if not _check_get_index_criteria(index.name):
                    continue
                index_obj = pc.Index(index.name)
                metadata = index_obj.fetch(ids=['db_metadata'])
                if metadata and metadata['vectors']['db_metadata']['metadata'].get('embedding_model') == embedding_model:
                    available_indexes.append(index.name)
                    index_metadatas.append(metadata['vectors']['db_metadata']['metadata'])
        try:
            if db_type == 'ChromaDB':
                _get_chromadb_indexes()
            elif db_type == 'Pinecone':
                _get_pinecone_indexes()
            elif db_type == 'RAGatouille':
                index_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille/colbert/indexes')
                if os.path.exists(index_path):
                    for item in os.listdir(index_path):
                        index_dir = os.path.join(index_path, item)
                        if os.path.isdir(index_dir) and os.listdir(index_dir):
                            available_indexes.append(item)
        except:
            return []
        
        return available_indexes, index_metadatas
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

    @staticmethod
    def _get_pinecone_status(self):
        """Get status of Pinecone indexes."""
        api_key = os.getenv('PINECONE_API_KEY')
        return self._cached_pinecone_status(api_key)
    
    @staticmethod
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

        return Chroma(
            client=client,
            collection_name=self.index_name,
            embedding_function=self.embedding_service.get_embeddings()
        )
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((PineconeApiException, NotFoundException)),
        reraise=True
    )
    def _init_pinecone(self, clear=False):
        """Initialize Pinecone."""
        # Check if index exists
        pinecone_client, _, ServerlessSpec = self._deps.get_db_deps()
        PineconeVectorStore, _, _, _ = self._deps.get_vectorstore_deps()

        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY')) # Need to use the native client since langchain can't upload a dummy embedding :(
        if clear:
            self.delete_index()
            # Wait until index is deleted
            while self.index_name in [idx.name for idx in pc.list_indexes()]:
                time.sleep(2)

        # Try to describe the index
        try:
            pc.describe_index(self.index_name)
        except NotFoundException:
            # Index doesn't exist, create it
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
            self.retriever = self.vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs=search_kwargs
            )
        elif self.db_type == 'RAGatouille':
            self.retriever = self.vectorstore.as_langchain_retriever(
                k=search_kwargs['k']
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    def _get_multivector_retriever(self, search_kwargs):
        """Get multi-vector retriever for Parent-Child or Summary RAG types."""
        # FIXME check that hashing/indexing with parent/child is working. Doesn't look like it is. See documents.py for chunking hashing.
        # TODO update to use dependency cache
        from langchain.storage import LocalFileStore
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        
        lfs = LocalFileStore(
            os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', self.index_name)
        )
        self.retriever = MultiVectorRetriever(
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
    def _store_index_metadata(self, chunking_result):
        """Store index metadata based on the database type."""
        # TODO update dependency cache
        from langchain_openai import OpenAIEmbeddings
        from langchain_voyageai import VoyageAIEmbeddings
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        from pinecone import Pinecone as pinecone_client
        from chromadb import PersistentClient

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
            index_metadata['embedding_family']= "OpenAI"
            index_metadata['embedding_model'] = self.embedding_service.model_name
        elif isinstance(self.embedding_service.get_embeddings(), VoyageAIEmbeddings):
            index_metadata['embedding_family'] = "Voyage"
            index_metadata['embedding_model'] = self.embedding_service.model_name
        elif isinstance(self.embedding_service.get_embeddings(), HuggingFaceInferenceAPIEmbeddings):
            index_metadata['embedding_family'] = "Hugging Face"
            index_metadata['embedding_model'] = self.embedding_service.model_name

        embedding_size = self.embedding_service.get_dimension()
        metadata_vector = [1e-5] * embedding_size

        if self.db_type == "Pinecone":
            # TODO see if I can do this with langchain vectorstore, problem is I need to make a dunmmy embedding.
            pc_native = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc_native.Index(self.index_name)
            index.upsert(vectors=[{
                'id': 'db_metadata',
                'values': metadata_vector,
                'metadata': index_metadata
            }])
        
        elif self.db_type == "ChromaDB":
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
            
            summary_model_name = doc_processor.llm_service.model_name
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
            if len(name) > 63:
                raise ValueError(f"The ChromaDB collection name must be less than 63 characters. Entry: {name}")
            if not name[0].isalnum() or not name[-1].isalnum():
                raise ValueError(f"The ChromaDB collection name must start and end with an alphanumeric character. Entry: {name}")
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                raise ValueError(f"The ChromaDB collection name can only contain alphanumeric characters, underscores, or hyphens. Entry: {name}")
            if ".." in name:
                raise ValueError(f"The ChromaDB collection name cannot contain two consecutive periods. Entry: {name}")

    def _upsert_docs(self, chunking_result, batch_size, show_progress=False, clear=False):
        """Upsert documents. Used for all RAG types. Clear only used for RAGatouille."""
        # TODO update to use dependency cache
        from langchain.storage import LocalFileStore
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from ..processing.documents import DocumentProcessor

        # if show_progress:  
        #     import streamlit as st
        #     progress_text = "Document indexing in progress..."
        #     my_bar = st.progress(0, text=progress_text)
        
        # Standard for pinecone and chroma
        if self.rag_type == 'Standard':
            for i in range(0, len(chunking_result.chunks), batch_size):
                batch = chunking_result.chunks[i:i + batch_size]
                batch_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in batch]
                
                if self.db_type == "Pinecone":
                    self.vectorstore.add_documents(documents=batch, ids=batch_ids, namespace=self.namespace)
                elif self.db_type == "ChromaDB":
                    self.vectorstore.add_documents(documents=batch, ids=batch_ids)
                elif self.db_type == "RAGatouille":
                    continue
                else:
                    raise NotImplementedError
                # if show_progress:
                #     progress_percentage = min(1.0, (i + batch_size) / len(chunking_result.chunks))
                #     my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
        
        # RAGatouille for all chunks at once
        if self.db_type == "RAGatouille":
            # Process all chunks at once for RAGatouille
            self.vectorstore.index(
                collection=[chunk.page_content for chunk in chunking_result.chunks],
                document_ids=[DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in chunking_result.chunks],
                index_name=self.index_name,
                overwrite_index=clear,
                split_documents=True,
                document_metadatas=[chunk.metadata for chunk in chunking_result.chunks]
            )
        
        # Parent-Child or Summary for pinecone and chroma
        if self.rag_type == 'Parent-Child' or self.rag_type == 'Summary':
            if self.db_type == 'RAGatouille':
                raise NotImplementedError("RAGatouille does not support Parent-Child or Summary RAG types")
            else:
                lfs_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', self.index_name)
                store = LocalFileStore(lfs_path)
                
                id_key = "doc_id"
                retriever = MultiVectorRetriever(vectorstore=self.vectorstore, byte_store=store, id_key=id_key)

                if self.rag_type == 'Parent-Child':
                    chunks = chunking_result.chunks
                    pages = chunking_result.pages['parent_chunks']
                    doc_ids = chunking_result.pages['doc_ids']
                elif self.rag_type == 'Summary':
                    chunks = chunking_result.summaries
                    pages = chunking_result.pages['docs']
                    doc_ids = chunking_result.pages['doc_ids']
                else:
                    raise NotImplementedError

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in batch]
                    
                    if self.db_type == "Pinecone":
                        self.vectorstore.add_documents(documents=batch, ids=batch_ids, namespace=self.namespace)
                    else:
                        self.vectorstore.add_documents(documents=batch, ids=batch_ids)
                    
                    # if show_progress:
                    #     progress_percentage = i / len(chunks)
                    #     my_bar.progress(progress_percentage, text=f'Document indexing in progress...{progress_percentage*100:.2f}%')
                
                # Index parent docs all at once
                retriever.docstore.mset(list(zip(doc_ids, pages)))
                self.retriever = retriever
        
        # if show_progress:
        #     my_bar.empty()