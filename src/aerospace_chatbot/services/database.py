"""Database service implementations."""

import os, shutil
import re
from pathlib import Path
import time
import json
import logging

from ..core.cache import Dependencies
from ..services.prompts import CLUSTER_LABEL
from pinecone.exceptions import NotFoundException, PineconeApiException

from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    wait_fixed
)

# TODO check if get_available_indexes works if there is an index with no metadata. Observed "no avaialble indexes" error when there was an index with no metadata and another with.

def pinecone_retry(func):
    """Decorator for Pinecone operations with retry and rate limiting."""
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2) + wait_exponential(multiplier=1, min=4, max=10),  # Increased delays
        reraise=True
    )(func)

class DatabaseService:
    """Handles database operations for different vector stores."""
    
    def __init__(self, db_type, index_name, rag_type, embedding_service, doc_type='document'):
        """
        Initialize DatabaseService.

        Args:
            db_type (str): Type of database ('Pinecone', 'ChromaDB', 'RAGatouille')
            index_name (str): Name of the index
            rag_type (str): Type of RAG ('Standard', 'Parent-Child', 'Summary')
            embedding_service (EmbeddingService): Embedding service instance
            doc_type (str, optional): Type of document ('document', 'question'). Defaults to 'document'.
        """
        self.db_type = db_type
        self.index_name = index_name
        self.rag_type = rag_type
        self.embedding_service = embedding_service
        self.doc_type = doc_type
        self.vectorstore = None
        self.retriever = None
        self.namespace = None
        self.logger = logging.getLogger(__name__)

    def initialize_database(self, namespace=None, clear=False):
        """
        Initialize and store database connection."""

        # Validate index name and RAG type
        self.logger.info(f"Validating index {self.index_name} and RAG type {self.rag_type}")
        self._validate_index()
        self.namespace = namespace

        # Check if LOCAL_DB_PATH environment variable exists
        if not os.getenv('LOCAL_DB_PATH'):
            raise ValueError("LOCAL_DB_PATH environment variable must be set")

        # Initialize the vectorstore based on database type
        if self.db_type == 'Pinecone':
            self._init_pinecone(clear=clear)
        elif self.db_type == 'ChromaDB':
            self._init_chromadb(clear=clear)
        elif self.db_type == 'RAGatouille':
            self._init_ragatouille(clear=clear)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def index_data(self, data, batch_size=100):
        """
        Index processed documents or questions. 
        If data is ChunkingResult type, chunked documents are processed.
        If data is a list of Documents, questions are processed. Questions are not chunked or processed, just validated and upserted.
        Database must be initialized before calling this method.
        """
        from ..processing.documents import ChunkingResult
        _, _, _, _, _, _, Document, _ = Dependencies.LLM.get_chain_utils()

        if not self.vectorstore:
            raise ValueError("Database not initialized. Call initialize_database() before indexing data.")

        if not (isinstance(data, ChunkingResult) or (isinstance(data, list) and all(isinstance(d, Document) for d in data))):
            raise TypeError("data must be either a ChunkingResult or a list of Documents")

        # Add index metadata, upsert docs
        if isinstance(data, ChunkingResult):
            if self.doc_type == 'document' and data.rag_type != self.rag_type:
                raise ValueError(f"RAG type mismatch: ChunkingResult has '{data.rag_type}' but DatabaseService has '{self.rag_type}'")
            self._store_index_metadata(data)    # Only store metadata if documents are being indexed
        self._upsert_data(data, batch_size)

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
    
    def copy_vectors(self, source_namespace, batch_size=100):
        """Copies vectors from a source Pinecone namespace into the namespace assigned to the DatabaseService instance."""
        pinecone_client, _, _, _, _ = Dependencies.Storage.get_db_clients()

        if self.db_type != 'Pinecone':
            raise ValueError("Vector copying is only supported for Pinecone databases")
            
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(self.index_name)
        
        # Verify source namespace exists and get vector count
        stats = index.describe_index_stats()
        source_ns_key = "" if source_namespace is None else source_namespace
        if source_ns_key not in stats['namespaces']:
            raise ValueError(f"Source namespace '{source_namespace}' does not exist in index {self.index_name}")
        
        if source_namespace is None:
            source_count = stats['namespaces']['']['vector_count']
        else:
            source_count = stats['namespaces'][source_namespace]['vector_count']

        if self.namespace is None:
            current_count = stats['namespaces']['']['vector_count']
        else:
            current_count = stats['namespaces'][self.namespace]['vector_count']
        expected_count = source_count + current_count
        self.logger.info(f"Expected count after copy into {self.namespace}: {expected_count}")
        
        # Get list of all vector IDs
        ids = []
        for id_batch in index.list(namespace=source_namespace):
            ids.extend(id_batch)
        self.logger.info(f"Copying {len(ids)} vectors from {source_namespace} to {self.namespace}")
            
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
            index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)

        # Verify the upload was successful
        self._verify_pinecone_upload(index, expected_count)
        self.logger.info(f"Copied {len(ids)} vectors from {source_namespace} to {self.namespace}")

    def delete_index(self):
        """Delete an index from the database."""
        
        def _delete_chromadb():
            """Delete ChromaDB index."""
            _, _, _, PersistentClient, _ = Dependencies.Storage.get_db_clients()
            
            try:
                db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
                client = PersistentClient(path=str(db_path))
                
                collections = client.list_collections()
                if any(c.name == self.index_name for c in collections):
                    client.delete_collection(self.index_name)
                else:
                    self.logger.warning(f"Collection {self.index_name} not found. No deletion performed.")
            except Exception as e:
                self.logger.warning(f"Failed to delete index {self.index_name}: {str(e)}")

        @pinecone_retry
        def _delete_pinecone():
            """Delete Pinecone index."""
            pinecone_client, _, _, _, _ = Dependencies.Storage.get_db_clients()
            
            pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            try:
                if self.index_name in [idx.name for idx in pc.list_indexes()]:
                    pc.delete_index(self.index_name)
                    self.logger.info(f"Index {self.index_name} deleted")
                else:
                    self.logger.warning(f"Index {self.index_name} not found. No deletion performed.")
            except Exception as e:
                self.logger.warning(f"Failed to delete index {self.index_name}: {str(e)}")

        def _delete_ragatouille():
            """Delete RAGatouille index."""
            try:
                ragatouille_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille/colbert/indexes', self.index_name)
                shutil.rmtree(ragatouille_path)
            except Exception as e:
                self.logger.warning(f"Failed to delete index {self.index_name}: {str(e)}")

        # Delete local file store if using Parent-Child or Summary RAG
        if self.rag_type in ['Parent-Child', 'Summary']:
            try:
                lfs_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', self.index_name)
                if os.path.exists(lfs_path):
                    shutil.rmtree(lfs_path)
                    self.logger.info(f"Local file store for {self.index_name} deleted")
            except Exception as e:
                self.logger.warning(f"Failed to delete local file store for {self.index_name}: {str(e)}")

        # Delete vector store based on type
        if self.db_type == 'ChromaDB':
            _delete_chromadb()
        elif self.db_type == 'Pinecone':
            _delete_pinecone()
        elif self.db_type == 'RAGatouille':
            _delete_ragatouille()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _init_chromadb(self, clear=False):
        """Initialize ChromaDB."""
        _, _, _, PersistentClient, _ = Dependencies.Storage.get_db_clients()
        _, Chroma, _, _ = Dependencies.Storage.get_vector_stores()
        
        # Create ChromaDB directory
        db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
        os.makedirs(db_path, exist_ok=True)
        client = PersistentClient(path=db_path)
        
        if clear:
            self.delete_index()

        # Create local file store directory if needed
        if self.rag_type in ['Parent-Child', 'Summary']:
            lfs_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', self.index_name)
            if not os.path.exists(lfs_path):
                os.makedirs(lfs_path)
                self.logger.info(f"Created local file store directory at {lfs_path}")

        self.vectorstore = Chroma(
            client=client,
            collection_name=self.index_name,
            embedding_function=self.embedding_service.get_embeddings()
        )
        self.logger.info(f"ChromaDB index {self.index_name} initialized or created.")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Increased delays
        retry=retry_if_exception_type((PineconeApiException, NotFoundException)),
        reraise=True
    )
    def _init_pinecone(self, clear=False):
        """Initialize Pinecone."""
        pinecone_client, _, ServerlessSpec, _, _ = Dependencies.Storage.get_db_clients()
        PineconeVectorStore, _, _, _ = Dependencies.Storage.get_vector_stores()

        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY')) # Need to use the native client since langchain can't upload a dummy embedding :(
        if clear:
            self.logger.info(f"Clearing Pinecone index {self.index_name}")
            self.delete_index()
            # Wait until index is deleted with timeout
            start_time = time.time()
            timeout = 30
            while self.index_name in [idx.name for idx in pc.list_indexes()]:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for index {self.index_name} to be deleted")
                time.sleep(4)

        # Try to describe the index
        try:
            pc.describe_index(self.index_name)
            self.logger.info(f"Pinecone index {self.index_name} found, not creating. Will be initialized with existing index.")
        except NotFoundException:
            # Index doesn't exist, create it
            self.logger.info(f"Not clearing database, but pinecone index {self.index_name} not found, creating.")
            pc.create_index(
                self.index_name,
                dimension=self.embedding_service.get_dimension(),
                spec=ServerlessSpec(
                    cloud='aws',    
                    region='us-west-2'
                )
            )
            self.logger.info(f"Pinecone index {self.index_name} created")
        
        # Create local file store directory if needed
        if self.rag_type in ['Parent-Child', 'Summary']:
            lfs_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', self.index_name)
            if not os.path.exists(lfs_path):
                os.makedirs(lfs_path)
                self.logger.info(f"Created local file store directory at {lfs_path}")

        self.vectorstore = PineconeVectorStore(
            index=pc.Index(self.index_name),
            index_name=self.index_name,
            embedding=self.embedding_service.get_embeddings(),
            text_key='page_content',
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            namespace=self.namespace
        )

    def _init_ragatouille(self, clear=False):
        """Initialize RAGatouille."""
        _, _, _, _, RAGPretrainedModel = Dependencies.Storage.get_db_clients()

        # FIXME this will always create a new index
        index_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille')
        if clear:
            self.delete_index()
            self.vectorstore = RAGPretrainedModel.from_pretrained(
                pretrained_model_name_or_path=self.embedding_service.model,
                index_root=index_path
            )
        else:
            self.vectorstore = RAGPretrainedModel.from_index(
                index_path=os.path.join(index_path, 'colbert/indexes', 
                                        self.index_name)
            )
        self.logger.info(f"RAGatouille index {self.index_name} initialized")
    
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
        _, _, MultiVectorRetriever, LocalFileStore = Dependencies.Storage.get_vector_stores()
        
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
        OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = Dependencies.Embeddings.get_models()
        pinecone_client, _, _, PersistentClient, _ = Dependencies.Storage.get_db_clients()

        index_metadata = {}
        if chunking_result.merge_pages is not None:
            index_metadata['merge_pages'] = chunking_result.merge_pages
        if chunking_result.chunk_method != 'None':
            index_metadata['chunk_method'] = chunking_result.chunk_method
        if chunking_result.chunk_size is not None:
            index_metadata['chunk_size'] = chunking_result.chunk_size
        if chunking_result.chunk_overlap is not None:
            index_metadata['chunk_overlap'] = chunking_result.chunk_overlap
        if isinstance(self.embedding_service.get_embeddings(), OpenAIEmbeddings):
            index_metadata['embedding_family']= "OpenAI"
            index_metadata['embedding_model'] = self.embedding_service.model
        elif isinstance(self.embedding_service.get_embeddings(), VoyageAIEmbeddings):
            index_metadata['embedding_family'] = "Voyage"
            index_metadata['embedding_model'] = self.embedding_service.model
        elif isinstance(self.embedding_service.get_embeddings(), HuggingFaceInferenceAPIEmbeddings):
            index_metadata['embedding_family'] = "Hugging Face"
            index_metadata['embedding_model'] = self.embedding_service.model

        embedding_size = self.embedding_service.get_dimension()
        metadata_vector = [1e-5] * embedding_size

        @pinecone_retry
        def _store_pinecone_metadata():
            pc_native = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc_native.Index(self.index_name)
            
            # Check if metadata vector already exists
            existing_metadata = index.fetch(ids=['db_metadata'])
            if 'db_metadata' in existing_metadata['vectors']:
                # Extract existing metadata
                db_metadata = existing_metadata['vectors']['db_metadata']['metadata']
                self.logger.info(f"Found existing metadata in {self.index_name}: {db_metadata}")
                
                # Check for mismatches between existing and new metadata
                mismatched_keys = []
                for key, value in db_metadata.items():
                    if key in index_metadata and index_metadata[key] != value:
                        mismatched_keys.append(f"{key}: expected '{value}', found '{index_metadata[key]}'")
                
                if mismatched_keys:
                    raise ValueError(f"Metadata mismatch in {self.index_name}. Mismatched values: {', '.join(mismatched_keys)}")
                
                self.logger.warning(f"Metadata vector already exists in {self.index_name}, skipping metadata upsert")
                return
            
            # Proceed with metadata upsert if none exists
            self.logger.info(f"No existing metadata found in {self.index_name}, adding metadata: {index_metadata}")
            index.upsert(vectors=[{
                'id': 'db_metadata',
                'values': metadata_vector,
                'metadata': index_metadata
            }])
            self._verify_pinecone_upload(index, 1)
            self.logger.info(f"Successfully added metadata to {self.index_name}")

        if self.db_type == "Pinecone":
            _store_pinecone_metadata()
        elif self.db_type == "ChromaDB":
            chroma_native = PersistentClient(path=os.path.join(os.getenv('LOCAL_DB_PATH'),'chromadb'))    
            index = chroma_native.get_collection(name=self.index_name)
            
            # Get existing metadata
            existing_metadata = index.get(ids=['db_metadata'])
            
            # If metadata exists, check for mismatches
            if existing_metadata['ids']:
                # Extract existing metadata
                db_metadata = existing_metadata['metadatas'][0]
                self.logger.info(f"Found existing metadata in {self.index_name}: {db_metadata}")
                
                # Check for mismatches between existing and new metadata
                mismatched_keys = []
                for key, value in db_metadata.items():
                    if key in index_metadata and index_metadata[key] != value:
                        mismatched_keys.append(f"{key}: expected '{value}', found '{index_metadata[key]}'")
                
                if mismatched_keys:
                    raise ValueError(f"Metadata mismatch in {self.index_name}. Mismatched values: {', '.join(mismatched_keys)}")
                
                self.logger.warning(f"Metadata vector already exists in {self.index_name}, skipping metadata upsert")
                return
            
            # Proceed with metadata upsert if none exists
            self.logger.info(f"No existing metadata found in {self.index_name}, adding metadata: {index_metadata}")
            index.add(
                embeddings=[metadata_vector],
                metadatas=[index_metadata],
                ids=['db_metadata']
            )
            self.logger.info(f"Successfully added metadata to {self.index_name}")

        elif self.db_type == "RAGatouille":
            self.logger.warning("Metadata storage is not yet supported for RAGatouille indexes")

    def _validate_index(self):
        """
        Validate and format index with parameters.
        Does nothing if no exceptions are raised, unless:
          RAG type is Parent-Child or Summary, which will append to index_name.
          doc_type is question, which will append 'queries' to index_name.
        """
        if not self.index_name or not self.index_name.strip():
            raise ValueError("Index name cannot be empty or contain only whitespace")

        # Clean the base name
        name = self.index_name.lower().strip()
        
        # Add RAG type suffix
        if self.doc_type == 'question':
            name += '-queries'
            self.index_name = name
            self.logger.info(f"Adding query type suffix to index name: {name}")
        elif self.doc_type == 'document':
            if self.rag_type == 'Parent-Child':
                name += '-parent-child'
                self.index_name = name
                self.logger.info(f"Adding RAG type suffix to index name: {name}")
            elif self.rag_type == 'Summary':
                name += '-summary'
                self.index_name = name
                self.logger.info(f"Adding RAG type suffix to index name: {name}")

        # Database-specific validation
        if self.db_type == "Pinecone":
            if len(name) > 45:
                raise ValueError(f"The Pinecone index name must be less than 45 characters. Entry: {name}")
            if '_' in name:
                raise ValueError(f"The Pinecone index name cannot contain underscores. Entry: {name}")
        elif self.db_type == "ChromaDB":
            if len(name) > 63:
                raise ValueError(f"The ChromaDB collection name must be less than 63 characters. Entry: {name}")
            if not name[0].isalnum() or not name[-1].isalnum():
                raise ValueError(f"The ChromaDB collection name must start and end with an alphanumeric character. Entry: {name}")
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                raise ValueError(f"The ChromaDB collection name can only contain alphanumeric characters, underscores, or hyphens. Entry: {name}")
            if ".." in name:
                raise ValueError(f"The ChromaDB collection name cannot contain two consecutive periods. Entry: {name}")

    def _upsert_data(self, upsert_data, batch_size):
        """Upsert documents or questions. Used for all RAG types."""
        from ..processing.documents import DocumentProcessor, ChunkingResult
        _, _, _, _, _, _, Document, _ = Dependencies.LLM.get_chain_utils()
        
        @pinecone_retry
        def _upsert_pinecone(batch, batch_ids):
            self.vectorstore.add_documents(documents=batch, ids=batch_ids, namespace=self.namespace)

        if self.db_type == "Pinecone":
            pinecone_client, _, _, _, _ = Dependencies.Storage.get_db_clients()

            pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))            
            pc_index = pc.Index(self.index_name)
            stats = pc_index.describe_index_stats()
            initial_count = stats['total_vector_count']
            if self.namespace:
                initial_count = stats['namespaces'].get(self.namespace, {}).get('vector_count', 0)   
            self.logger.info(f"Initial vector count: {initial_count}")

        if isinstance(upsert_data, ChunkingResult):
            # Chunked documents
            # Standard for pinecone and chroma
            if self.rag_type == 'Standard':
                total_batches = (len(upsert_data.chunks) + batch_size - 1) // batch_size  # Calculate total number of batches
                for i in range(0, len(upsert_data.chunks), batch_size):
                    current_batch = (i // batch_size) + 1  # Calculate current batch number (1-based index)
                    self.logger.info(f"Upserting batch {current_batch} of {total_batches}")
                    batch = upsert_data.chunks[i:i + batch_size]
                    batch_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in batch]
                    
                    if self.db_type == "Pinecone":
                        _upsert_pinecone(batch, batch_ids)
                    elif self.db_type == "ChromaDB":
                        self.vectorstore.add_documents(documents=batch, ids=batch_ids)
                    elif self.db_type == "RAGatouille":
                        continue
                    else:
                        raise NotImplementedError
            
            # RAGatouille for all chunks at once
            if self.db_type == "RAGatouille":
                # Process all chunks at once for RAGatouille
                self.vectorstore.index(
                    collection=[chunk.page_content for chunk in upsert_data.chunks],
                    document_ids=[DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in upsert_data.chunks],
                    index_name=self.index_name,
                    split_documents=True,
                    document_metadatas=[chunk.metadata for chunk in upsert_data.chunks]
                )
            
            # Parent-Child or Summary for pinecone and chroma
            if self.rag_type in ['Parent-Child', 'Summary']:
                _, _, MultiVectorRetriever, LocalFileStore = Dependencies.Storage.get_vector_stores()
                
                if self.db_type == 'RAGatouille':
                    raise NotImplementedError("RAGatouille does not support Parent-Child or Summary RAG types")
                else:
                    lfs_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', self.index_name)
                    if not os.path.exists(lfs_path):
                        raise ValueError(f"Local file store path {lfs_path} does not exist. This should have been created during database initialization.")
                    store = LocalFileStore(lfs_path)
                    
                    id_key = "doc_id"
                    retriever = MultiVectorRetriever(vectorstore=self.vectorstore, byte_store=store, id_key=id_key)

                    # Get the appropriate chunks and pages based on RAG type
                    if self.rag_type == 'Parent-Child':
                        chunks = upsert_data.chunks
                        pages = upsert_data.pages['parent_chunks']
                        doc_ids = upsert_data.pages['doc_ids']
                    elif self.rag_type == 'Summary':
                        chunks = upsert_data.summaries
                        pages = upsert_data.pages['docs']
                        doc_ids = upsert_data.pages['doc_ids']
                    else:
                        raise NotImplementedError

                    # Process chunks in batches
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        batch_ids = [DocumentProcessor.stable_hash_meta(chunk.metadata) for chunk in batch]
                        
                        if self.db_type == "Pinecone":
                            _upsert_pinecone(batch, batch_ids)
                        else:
                            self.vectorstore.add_documents(documents=batch, ids=batch_ids)
                    
                    # Index parent docs all at once
                    retriever.docstore.mset(list(zip(doc_ids, pages)))
                    self.retriever = retriever
        elif isinstance(upsert_data, list) and all(isinstance(item, Document) for item in upsert_data):
            # Questions
            if self.db_type in ["Pinecone", "ChromaDB"]:
                self.vectorstore.add_documents(documents=upsert_data)
            elif self.db_type == "RAGatouille":
                self.logger.warning("RAGatouille does not support question databases.")
            else:
                raise NotImplementedError
        else:
            raise ValueError("Unsupported data formatting for upsert. Should either be a ChunkingResult or a list of Documents.")
            
        if self.db_type == "Pinecone":
            if isinstance(upsert_data, ChunkingResult):
                expected_count = initial_count + len(upsert_data.summaries if self.rag_type == 'Summary' else upsert_data.chunks)
            else:
                expected_count = initial_count+len(upsert_data)
            self._verify_pinecone_upload(pc_index, expected_count)
        
    def _verify_pinecone_upload(self, index, expected_count):
        """Verify that vectors were uploaded to Pinecone index."""
        max_retries = 15  # Increased from 10
        retry_count = 0
        # last_count = 0
        # stable_count_checks = 0
        
        while retry_count < max_retries:
            time.sleep(5)   # Added delay to reduce likelihood of overlap with delayed delete responses
            stats = index.describe_index_stats()
            current_count = stats['total_vector_count']
            if self.namespace:
                current_count = stats['namespaces'].get(self.namespace, {}).get('vector_count', 0)
                
            # Check if count has stabilized
            # if current_count == last_count:
            #     stable_count_checks += 1
            #     if stable_count_checks >= 3:  # Count has been stable for 3 checks
            #         if current_count > expected_count:
            #             self.logger.warning(f"Final count ({current_count}) exceeds expected count ({expected_count})")
            #         self.logger.info(f"Vector count has stabilized at {current_count}")
            #         break
            # else:
            #     stable_count_checks = 0
                
            # TODO explore making this a check of the exact count. I found that the count was sometimes greater than expected due to latency in the Pinecone API.
            if current_count >= expected_count:
                self.logger.info(f"Successfully verified {current_count} vectors in Pinecone index{f' namespace {self.namespace}' if self.namespace else ''}")
                break
                
            self.logger.info(f"Waiting for vectors to be indexed in Pinecone... Current count: {current_count}, Expected: {expected_count}")
            time.sleep(4) 
            retry_count += 1
            
        if retry_count == max_retries:
            if current_count > expected_count:
                self.logger.warning(f"Final count ({current_count}) exceeds expected count ({expected_count})")
                return  # Don't raise error if we have more vectors than expected
            raise TimeoutError(f"Timeout waiting for vectors to be indexed in Pinecone. Current count: {current_count}, Expected: {expected_count}")

def get_docs_questions_df(db_service, query_db_service, logger=False):
    """Get documents and questions from database as a DataFrame."""
    pd, _, _, _ = Dependencies.Analysis.get_tools()
    pinecone_client, _, _, _, _ = Dependencies.Storage.get_db_clients()
    
    @pinecone_retry
    def _fetch_pinecone_docs(index_name):
        """Fetch documents from Pinecone in batches with retry logic and rate limiting."""
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(index_name)

        if logger:
            logger.info(f'Attempting to fetch {index_name} from Pinecone')

        ids = []
        for id_batch in index.list():
            ids.extend(id_batch)
            
        vectors_temp = []
        docs = []
        chunk_size = 200
        for i in range(0, len(ids), chunk_size):
            vector=index.fetch(ids[i:i+chunk_size])['vectors']
            vector_data = []
            for _, value in vector.items():
                vector_data.append(value)
            vectors_temp.extend(vector)
            docs.extend(vector_data)

        if logger:
            logger.info(f'Retrieved {len(docs)} documents')
        return docs

    def _process_chromadb_response(response, doc_type):
        """Process ChromaDB response into DataFrame."""

        if doc_type == 'document':
            # Filter out db_metadata document
            filtered_indices = [i for i, id in enumerate(response["ids"]) if id != 'db_metadata']
            
            return pd.DataFrame({
                "id": [response["ids"][i] for i in filtered_indices],
                "source": [response["metadatas"][i].get("source", "") for i in filtered_indices],
                "page": [int(response["metadatas"][i].get("page", -1)) for i in filtered_indices],
                "metadata": [response["metadatas"][i] for i in filtered_indices],
                "document": [response["documents"][i] for i in filtered_indices],
                "embedding": [response["embeddings"][i] for i in filtered_indices],
            })
        else:  # question
            return pd.DataFrame({
                "id": response["ids"],
                "question": response["documents"],
                "answer": [metadata.get("answer", "") for metadata in response["metadatas"]],
                "sources": [metadata.get("sources", "").split(",") if metadata.get("sources") else [] for metadata in response["metadatas"]],
                "embedding": response["embeddings"],
            })

    def _process_pinecone_response(docs, doc_type):
        """Process Pinecone response into DataFrame."""
        if doc_type == 'document':
            # Skip the db_metadata document
            filtered_docs = [doc for doc in docs if doc['id'] != 'db_metadata']
            
            return pd.DataFrame({
                "id": [data['id'] for data in filtered_docs],
                "source": [data['metadata'].get('source', '') for data in filtered_docs],
                "page": [int(data['metadata'].get('page', -1)) for data in filtered_docs],
                "metadata": [data['metadata'] for data in filtered_docs],
                "document": [data['metadata'].get('page_content', '') for data in filtered_docs],
                "embedding": [data['values'] for data in filtered_docs],
            })
        else:  # question
            return pd.DataFrame({
                "id": [data['id'] for data in docs],
                "question": [data['metadata'].get('page_content', '') for data in docs],
                "answer": [data['metadata'].get('answer', '') for data in docs],
                "sources": [data['metadata'].get('sources', '').split(',') if data['metadata'].get('sources') else [] for data in docs],
                "embedding": [data['values'] for data in docs],
            })

    def _add_rag_columns(df, index_name):
        """Add RAG-specific columns to document DataFrame if needed."""
        rag_type = 'Standard'
        if index_name.endswith('-parent-child'):
            rag_type = 'Parent-Child'
        elif index_name.endswith('-summary'):
            rag_type = 'Summary'
            
        if rag_type != 'Standard':
            json_data_list = []
            for i, row in df.iterrows():
                doc_id = row['metadata']['doc_id']
                file_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', index_name, f"{doc_id}")
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                json_data = json_data['kwargs']['page_content']
                json_data_list.append(json_data)
                
            column_name = 'parent-doc' if rag_type == 'Parent-Child' else 'original-doc'
            df[column_name] = json_data_list
        return df

    def _get_dataframe_from_store(vectorstore, db_type, index_name, doc_type='document', logger=False):
        """Common method to retrieve data from any vectorstore type and convert to DataFrame."""
        # Validate doc_type
        if doc_type not in ('document', 'question'):
            raise ValueError(f"Invalid doc_type: {doc_type}. Must be either 'document' or 'question'")

        if db_type == 'ChromaDB':
            response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
            df = _process_chromadb_response(response, doc_type)
        elif db_type == 'Pinecone':
            docs = _fetch_pinecone_docs(index_name)
            df = _process_pinecone_response(docs, doc_type)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        if doc_type == 'question' and logger:
            logger.info(f'Retrieved {len(docs)} questions')

        # Add RAG-specific columns if needed
        if doc_type == 'document':
            df = _add_rag_columns(df, index_name)

        return df

    # Check if database exists
    db_status = get_database_status(db_service.db_type)
    if not db_status['status']:
        raise Exception('Unable to access database')
        
    if db_service.db_type == 'ChromaDB':
        collections = [collection.name for collection in db_status['message']]
    elif db_service.db_type == 'Pinecone':
        collections = db_status['message']  # Already a list of index names
    else:
        raise ValueError(f"Unsupported database type: {db_service.db_type}. RAGatouille not supported.")
    
    matching_collection = [collection for collection in collections if collection == query_db_service.index_name]
    if len(matching_collection) > 1:
        raise Exception('Multiple matching query collections found.')
    if not matching_collection:
        raise Exception('Query database not found. Please create a query database using the Chatbot page and a selected index.')

    # Get documents and questions dataframes
    docs_df = _get_dataframe_from_store(
        db_service.vectorstore,
        db_service.db_type,
        db_service.index_name,
        doc_type='document'
    )
    docs_df["type"] = "doc"

    questions_df = _get_dataframe_from_store(
        query_db_service.vectorstore,
        query_db_service.db_type,
        query_db_service.index_name,
        doc_type='question'
    )
    questions_df["type"] = "question"

    # Process relationships
    questions_df["num_sources"] = questions_df["sources"].apply(len)
    questions_df["first_source"] = questions_df["sources"].apply(
        lambda x: next(iter(x), None)
    )

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

    return pd.concat([docs_df, questions_df], ignore_index=True)

def add_clusters(df, n_clusters, llm_service=None, docs_per_cluster: int = 10):
    """Add cluster labels to DataFrame using KMeans clustering."""
    _, np, KMeans, _ = Dependencies.Analysis.get_tools()
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(np.stack(df['embedding'].values))
    
    # Generate cluster labels using LLM if provided
    if llm_service is not None:
        cluster_labels = []
        for i in range(n_clusters):
            cluster_docs = df[df['cluster'] == i]['document'].head(docs_per_cluster).tolist()
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

def export_to_hf_dataset(df, dataset_name):
    """Export DataFrame to Hugging Face dataset."""
    _, _, _, Dataset = Dependencies.Analysis.get_tools()
    
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(
        dataset_name, 
        token=os.getenv('HUGGINGFACEHUB_API_KEY')
    )

def get_available_indexes(db_type, embedding_model=None, rag_type=None):
    """Get available indexes based on current settings. If embedding_model or rag_type are None, 
    returns all indexes without filtering on those criteria."""
    logger = logging.getLogger(__name__)

    def _check_get_index_criteria(index_name, rag_type):
        """Check if index meets criteria for inclusion."""
        # Never include query indexes when filtering
        if index_name.endswith('-queries'):
            return False
            
        # Check RAG type criteria if specified
        if rag_type is not None:
            rag_matches = (
                (rag_type == 'Parent-Child' and index_name.endswith('-parent-child')) or
                (rag_type == 'Summary' and index_name.endswith('-summary')) or
                (rag_type == 'Standard' and not index_name.endswith(('-parent-child', '-summary')))
            )
            if not rag_matches:
                return False
                
        return True

    def _process_chromadb_indexes():
        """Process available ChromaDB indexes."""
        _, _, _, PersistentClient, _ = Dependencies.Storage.get_db_clients()

        client = PersistentClient(path=os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb'))
        
        available_indexes = []
        index_metadatas = []
        
        for index in db_status['message']:
            # First check if index meets basic criteria
            if not _check_get_index_criteria(index.name, rag_type):
                continue
                
            collection = client.get_collection(index.name)
            metadata = collection.get(ids=['db_metadata'])
            
            # Check embedding model if specified
            if metadata and (embedding_model is None or metadata['metadatas'][0].get('embedding_model') == embedding_model):
                available_indexes.append(index.name)
                index_metadatas.append(metadata['metadatas'][0])
                
        return available_indexes, index_metadatas

    def _process_pinecone_indexes():
        """Process available Pinecone indexes."""
        pinecone_client, _, _, _, _ = Dependencies.Storage.get_db_clients()
        
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        
        available_indexes = []
        index_metadatas = []
        
        for index_name in db_status['message']:
            # First check if index meets basic criteria
            if not _check_get_index_criteria(index_name, rag_type):
                continue
                
            index_obj = pc.Index(index_name)
            
            try:
                metadata = index_obj.fetch(ids=['db_metadata'])
                # Check embedding model if specified
                if metadata and (embedding_model is None or metadata['vectors']['db_metadata']['metadata'].get('embedding_model') == embedding_model):
                    available_indexes.append(index_name)
                    index_metadatas.append(metadata['vectors']['db_metadata']['metadata'])
            except Exception as e:
                # Log a message if metadata is not found
                logger.warning(f"Index {index_name} exists but does not have db_metadata vector. Skipping. {e}")
                
        return available_indexes, index_metadatas

    def _process_ragatouille_indexes():
        """Process available RAGatouille indexes."""
        available_indexes = []
        index_metadatas = []
        
        for index_name in db_status['message']:
            # First check if index meets basic criteria
            if not _check_get_index_criteria(index_name, rag_type,):
                continue
                
            available_indexes.append(index_name)
            # RAGatouille doesn't support metadata storage currently
            index_metadatas.append({})
            
        return available_indexes, index_metadatas

    # Get database status
    db_status = get_database_status(db_type)
    logger.info(f"Database status in get_available_indexes for {db_type}, {embedding_model}, {rag_type}: {db_status}")
    
    if not db_status['status']:
        return [], []
        
    try:
        # Get filtered list of indexes based on database type and criteria
        if db_type == 'ChromaDB':
            available_indexes, index_metadatas = _process_chromadb_indexes()
        elif db_type == 'Pinecone':
            available_indexes, index_metadatas = _process_pinecone_indexes()
        elif db_type == 'RAGatouille':
            available_indexes, index_metadatas = _process_ragatouille_indexes()
            
        return available_indexes, index_metadatas
        
    except Exception as e:
        logger.error(f"Error in get_available_indexes: {str(e)}")
        return [], []

def get_database_status(db_type):
    """Get status of database indexes/collections."""

    if db_type == 'Pinecone':
        return _get_pinecone_status()
    elif db_type == 'ChromaDB':
        return _get_chromadb_status()
    elif db_type == 'RAGatouille':
        return _get_ragatouille_status()
    else:
        return {'status': False, 'message': f'Unsupported database type: {db_type}'}

def _get_pinecone_status():
    """Get status of Pinecone indexes."""
    
    try:
        pinecone_client, _, _, _, _ = Dependencies.Storage.get_db_clients()

        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        indexes = pc.list_indexes()
        if not indexes:
            return {'status': False, 'message': 'No indexes found'}
        return {'status': True, 'message': [idx.name for idx in indexes]}
    except Exception as e:
        return {'status': False, 'message': f'Error connecting to Pinecone: {str(e)}'}

def _get_chromadb_status():
    """Get status of ChromaDB collections."""
    try:
        _, _, _, PersistentClient, _ = Dependencies.Storage.get_db_clients()
        
        db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
        client = PersistentClient(path=str(db_path))
        collections = client.list_collections()
        if not collections:
            return {'status': False, 'message': 'No collections found'}
        return {'status': True, 'message': collections}
    except Exception as e:
        return {'status': False, 'message': f'Error connecting to ChromaDB: {str(e)}'}

def _get_ragatouille_status():
    """Get status of RAGatouille indexes."""
    try:
        index_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille/colbert/indexes')
        if not os.path.exists(index_path):
            return {'status': False, 'message': 'No indexes found'}
        
        indexes = [item for item in os.listdir(index_path) 
                    if os.path.isdir(os.path.join(index_path, item))]
        if not indexes:
            return {'status': False, 'message': 'No indexes found'}
        return {'status': True, 'message': indexes}
    except Exception as e:
        return {'status': False, 'message': f'Error accessing RAGatouille indexes: {str(e)}'}