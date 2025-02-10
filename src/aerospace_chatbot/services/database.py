"""Database service implementations."""

import os, shutil, re, time, json, logging

# Databases
from pinecone import Pinecone as pinecone_client
from pinecone import ServerlessSpec
from pinecone.exceptions import NotFoundException, PineconeApiException
from ragatouille import RAGPretrainedModel
from langchain_pinecone import PineconeVectorStore

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import cohere

# Utilities
from langchain_core.documents import Document
from langchain_core.runnables import chain
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datasets import Dataset
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    wait_fixed
)

# Typing
from typing_extensions import List
from typing import List, Tuple

# Services
from ..services.prompts import CLUSTER_LABEL

# TODO check if get_available_indexes works if there is an index with no metadata. Observed "no avaialble indexes" error when there was an index with no metadata and another with.

class DatabaseService:
    """Handles database operations for different vector stores."""
    
    def __init__(self, db_type, index_name, embedding_service, rerank_service=None):
        """
        Initialize DatabaseService.
        """
        self.db_type = db_type
        self.index_name = index_name
        self.embedding_service = embedding_service
        self.rerank_service = rerank_service    # Optional, defaults to None. Exceptions raised if rerank_service is none if you you try to rerank.
        self.vectorstore = None
        self.retriever = None
        self.namespace = None
        self.db_client = None
        self.logger = logging.getLogger(__name__)

    def initialize_database(self, namespace=None, clear=False):
        """
        Initialize and store database connection.
        """
        # Validate index name and RAG type
        self.logger.info(f"Validating index {self.index_name}")
        self._validate_index()
        self.namespace = namespace

        # Check if LOCAL_DB_PATH environment variable exists
        if not os.getenv('LOCAL_DB_PATH'):
            raise ValueError("LOCAL_DB_PATH environment variable must be set")

        # Initialize the vectorstore based on database type
        if self.db_type == 'Pinecone':
            self._init_pinecone(clear=clear)
        elif self.db_type == 'RAGatouille':
            self._init_ragatouille(clear=clear)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def index_data(self, chunks, batch_size=100):
        """
        Index processed documents or questions. 
        Database must be initialized before calling this method.
        """
        if not self.vectorstore:
            raise ValueError("Database not initialized. Call initialize_database() before indexing data.")

        self._store_index_metadata(chunks)    # Only store metadata if documents are being indexed
        try:
            chunks.chunk_convert()
        except AttributeError as e:
            self.logger.warning(f"Chunks already converted into doc format: {str(e)}")
            pass
        self._upsert_data(chunks, batch_size)

    def get_retriever(self, k=8):
        """
        Get configured retriever for the vectorstore.
        """
        self.retriever = None
        search_kwargs = self._process_retriever_args(k)

        if not self.vectorstore:
            raise ValueError("Database not initialized. Please ensure database is initialized before getting retriever.")

        self._get_standard_retriever(search_kwargs)

    def rerank(self, query: str, retrieved_docs: List[Tuple[Document, float]], top_n: int = None):
        # TODO only works with cohere rerank, if others are used, consider using langchain reranker or updating the class.
        # retrieved_docs contains a list of tuples, where the first element is the document and the second is the score

        # Require top_n to be at least 3
        if top_n is None:
            top_n = 3
        if top_n < 3:
            raise ValueError("top_n must be at least 3")
        elif top_n > len(retrieved_docs):
            raise ValueError("top_n must be less than or equal to the number of retrieved documents")

        # Rerank docs
        if self.rerank_service is None:
            raise ValueError("Rerank service is not set. Please set rerank_service before calling this method.")
        else:
            rerank_results = self.rerank_service.rerank_docs(
                query=query,
                retrieved_docs=retrieved_docs,
                top_n=top_n
            )

        # Create a dictionary to map document IDs to rerank scores
        rerank_scores = {retrieved_docs[i][0].id: item.relevance_score for i, item in enumerate(rerank_results)}

        # Create list of (doc, original_score, rerank_score) tuples
        doc_scores = []
        for doc, original_score in retrieved_docs:
            rerank_score = rerank_scores.get(doc.id, None)  # Get the rerank score or None if not available
            doc_scores.append((doc, original_score, rerank_score))

        # Sort docs by rerank score in descending order, placing those without a rerank score at the end
        doc_scores_sorted = sorted(doc_scores, key=lambda x: (x[2] is not None, x[2]), reverse=True)

        return doc_scores_sorted
    
    def copy_vectors(self, source_namespace, batch_size=100):
        """
        Copies vectors from a source Pinecone namespace into the namespace assigned to the DatabaseService instance.
        """
        if self.db_type != 'Pinecone':
            raise ValueError("Vector copying is only supported for Pinecone databases")
            
        index = self.db_client.Index(self.index_name)
        
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
        @pinecone_retry
        def _delete_pinecone():
            """Delete Pinecone index."""
            # pinecone_client, _, _, _, _ = Dependencies.Storage.get_db_clients()
            
            try:
                if self.index_name in [idx.name for idx in self.db_client.list_indexes()]:
                    self.db_client.delete_index(self.index_name)
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

        if self.db_type == 'Pinecone':
            _delete_pinecone()
        elif self.db_type == 'RAGatouille':
            _delete_ragatouille()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Increased delays
        retry=retry_if_exception_type((PineconeApiException, NotFoundException)),
        reraise=True
    )
    def _init_pinecone(self, clear=False):
        """
        Initialize Pinecone.
        """
        self.db_client = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        if clear:
            self.logger.info(f"Clearing Pinecone index {self.index_name}")
            self.delete_index()
            # Wait until index is deleted with timeout
            start_time = time.time()
            timeout = 30
            while self.index_name in [idx.name for idx in self.db_client.list_indexes()]:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for index {self.index_name} to be deleted")
                time.sleep(4)

        # Try to describe the index
        try:
            self.db_client.describe_index(self.index_name)
            self.logger.info(f"Pinecone index {self.index_name} found, not creating. Will be initialized with existing index.")
        except NotFoundException:
            # Index doesn't exist, create it
            self.logger.info(f"Not clearing database, but pinecone index {self.index_name} not found, creating.")
            self.db_client.create_index(
                self.index_name,
                dimension=self.embedding_service.get_dimension(),
                spec=ServerlessSpec(
                    cloud='aws',    
                    region='us-west-2'
                )
            )
            self.logger.info(f"Pinecone index {self.index_name} created")

        self.vectorstore = PineconeVectorStore(
            index=self.db_client.Index(self.index_name),
            index_name=self.index_name,
            embedding=self.embedding_service.get_embeddings(),
            text_key='page_content',
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            namespace=self.namespace
        )

    def _init_ragatouille(self, clear=False):
        """
        Initialize RAGatouille.
        """
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
        """
        Get standard retriever based on index type.
        Returns a list of (doc, score) tuples sorted by score.
        """
        if self.db_type == 'Pinecone':
            # Callable retriever which adds score to the metadata of the documents
            @chain
            def retriever(query: str):
                # Convert to a list from zip tuple
                docs, scores = map(
                    list, 
                    zip(*self.vectorstore.similarity_search_with_score(
                        query, 
                        **search_kwargs
                    ))
                )
                
                # Create list of (doc, score) tuples and sort by score
                doc_scores = list(zip(docs, scores))
                doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)
                
                return doc_scores_sorted
            self.retriever = retriever
        elif self.db_type == 'RAGatouille':
            @chain
            def retriever(query: str):
                # Get raw search results
                raw_results = self.vectorstore.search(
                    query=query,
                    k=search_kwargs['k']
                )
                
                # Find max score for normalization
                max_score = max(result['score'] for result in raw_results)
                
                # Convert results to LangChain document format
                docs = []
                scores = []
                for result in raw_results:
                    # Create metadata dict combining document_metadata and top-level metadata
                    metadata = {
                        **result.get('document_metadata', {}),
                        'passage_id': result.get('passage_id'),
                        'rank': result.get('rank')
                    }
                    
                    # Create LangChain document
                    doc = Document(
                        id=result.get('document_id'),
                        page_content=result['content'],
                        metadata=metadata
                    )
                    
                    # Normalize score
                    normalized_score = result['score'] / max_score
                    
                    docs.append(doc)
                    scores.append(normalized_score)
                
                # Create list of (doc, score) tuples and sort by score
                doc_scores = list(zip(docs, scores))
                doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)
                
                return doc_scores_sorted
                
            self.retriever = retriever
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _process_retriever_args(self, k=8):
        """
        Process the retriever arguments.
        """
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
    def _store_index_metadata(self, chunks):
        """
        Store index metadata based on the database type. 
        This only works for Pinecone. RAGatouille does not support metadata vectors and will allow any chunk type to be upserted.
        """

        index_metadata = {}
        if chunks.chunk_size is not None:
            index_metadata['chunk_size'] = chunks.chunk_size
        if chunks.chunk_overlap is not None:
            index_metadata['chunk_overlap'] = chunks.chunk_overlap
        if self.db_type == 'RAGatouille':
            pass    # Do nothing because, metadata vector not used
        elif isinstance(self.embedding_service.get_embeddings(), OpenAIEmbeddings):
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
            index = self.db_client.Index(self.index_name)
            
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
        elif self.db_type == "RAGatouille":
            self.logger.warning("Metadata storage is not yet supported for RAGatouille indexes. Any chunk type will be upserted, regardless of existing metadata.")
        
    def _validate_index(self):
        """
        Validate and format index with parameters.
        Does nothing if no exceptions are raised.
        """
        if not self.index_name or not self.index_name.strip():
            raise ValueError("Index name cannot be empty or contain only whitespace")
        name = self.index_name.lower().strip()  # Clean the base name

        # Database-specific validation
        if len(name) > 45:
            raise ValueError(f"The index name must be less than 45 characters. Entry: {name}")
        if '_' in name:
            raise ValueError(f"The index name cannot contain underscores. Entry: {name}")

    def _upsert_data(self, upsert_data, batch_size):
        """
        Upsert documents or questions. Used for all RAG types.
        """
        def _upsert_pinecone(batch, batch_ids):
            # TODO the upsert validation is being cagey. Eventually, the upsert quantity looks consistent, but it's not deterministic when to check.
            # Get fresh count before upserting
            # stats = pc_index.describe_index_stats()
            # current_count = stats['total_vector_count']
            # if self.namespace:
            #     current_count = stats['namespaces'].get(self.namespace, {}).get('vector_count', 0)
            
            # self.logger.info(f"Current vector count before batch: {current_count}")
            self.logger.info(f"Upserting {len(batch)} vectors for this batch...")
            
            self.vectorstore.add_documents(documents=batch, ids=batch_ids, namespace=self.namespace)
            time.sleep(1)   # Mitigate rate limit/batch issues
            
            # Verify this batch was uploaded successfully
            # expected_batch_count = current_count + len(batch)
            # self._verify_pinecone_upload(pc_index, expected_batch_count)

        if self.db_type == "Pinecone":
            pc_index = self.db_client.Index(self.index_name)
            stats = pc_index.describe_index_stats()
            initial_count = stats['total_vector_count']
            if self.namespace:
                initial_count = stats['namespaces'].get(self.namespace, {}).get('vector_count', 0)   
            self.logger.info(f"Initial vector count: {initial_count}")
            self.logger.info(f"Upserting {len(upsert_data.chunks)} vectors")

            total_batches = (len(upsert_data.chunks) + batch_size - 1) // batch_size
            for i in range(0, len(upsert_data.chunks), batch_size):
                current_batch = (i // batch_size) + 1
                self.logger.info(f"Upserting batch {current_batch} of {total_batches}")
                batch = upsert_data.chunks[i:i + batch_size]
                batch_ids = [chunk.metadata['element_id'] for chunk in batch]
                
                if self.db_type == "Pinecone":
                    # Update current_count after each successful batch
                    _upsert_pinecone(batch, batch_ids)
                elif self.db_type == "RAGatouille":
                    continue
                else:
                    raise NotImplementedError
        
        # RAGatouille for all chunks at once
        if self.db_type == "RAGatouille":
            # Process all chunks at once for RAGatouille
            self.vectorstore.index(
                collection=[chunk.page_content for chunk in upsert_data.chunks],
                document_ids=[chunk.metadata['element_id'] for chunk in upsert_data.chunks],
                index_name=self.index_name,
                split_documents=True,
                document_metadatas=[chunk.metadata for chunk in upsert_data.chunks]
            )
        
    def _verify_pinecone_upload(self, index, expected_count):
        """
        Verify that vectors were uploaded to Pinecone index.
        """
        max_retries = 15
        retry_count = 0
        
        while retry_count < max_retries:
            time.sleep(2.5)   # Added delay to reduce likelihood of overlap with delayed delete responses
            stats = index.describe_index_stats()
            current_count = stats['total_vector_count']
            if self.namespace:
                current_count = stats['namespaces'].get(self.namespace, {}).get('vector_count', 0)
                
            # TODO explore making this a check of the exact count. I found that the count was sometimes greater than expected due to latency in the Pinecone API.
            if current_count >= expected_count:
                self.logger.info(f"Successfully verified {current_count} vectors in Pinecone index{f' namespace {self.namespace}' if self.namespace else ''}")
                break
                
            self.logger.info(f"Retry {retry_count + 1} of {max_retries}: Waiting for vectors to be indexed in Pinecone... Current count: {current_count}, Expected: {expected_count}")
            time.sleep(5) 
            retry_count += 1
            
        if retry_count == max_retries:
            if current_count > expected_count:
                self.logger.warning(f"Final count ({current_count}) exceeds expected count ({expected_count})")
                return  # Don't raise error if we have more vectors than expected
            raise TimeoutError(f"Timeout waiting for vectors to be indexed in Pinecone. Current count: {current_count}, Expected: {expected_count}")

def pinecone_retry(func):
    """
    Decorator for Pinecone operations with retry and rate limiting.
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2) + wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )(func)

def get_docs_df(db_service, logger=False):
    """
    Get documents from database as a DataFrame.
    # TODO add back in question mapping for future work
    """
    @pinecone_retry
    def _fetch_pinecone_docs(index_name):
        """Fetch documents from Pinecone in batches with retry logic and rate limiting."""
        index = db_service.db_client.Index(index_name)

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

    def _process_pinecone_response(docs):
        """
        Process Pinecone response into DataFrame.
        """
        return pd.DataFrame({
            "id": [data['id'] for data in docs],
            "page_content": [data['metadata'].get('page_content', '') for data in docs],
            "page_number": [data['metadata'].get('page_number', '') for data in docs],
            "file_name": [data['metadata'].get('file_name', '') for data in docs],
            "embedding": [data['values'] for data in docs],
        })

    # def _add_rag_columns(df, index_name):
    #     """
    #     Add RAG-specific columns to document DataFrame if needed.
    #     """
    #     json_data_list = []
    #     for i, row in df.iterrows():
    #         doc_id = row['metadata']['doc_id']
    #         file_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'local_file_store', index_name, f"{doc_id}")
    #         with open(file_path, "r") as f:
    #             json_data = json.load(f)
    #         json_data = json_data['kwargs']['page_content']
    #         json_data_list.append(json_data)
            
    #     df['original-doc'] = json_data_list
    #     return df

    def _get_dataframe_from_store(db_type, index_name):
        """
        Common method to retrieve data from any vectorstore type and convert to DataFrame.
        """
        if db_type == 'Pinecone':
            docs = _fetch_pinecone_docs(index_name)
            df = _process_pinecone_response(docs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        # df = _add_rag_columns(df, index_name)

        return df

    # Check if database exists
    db_status = get_database_status(db_service.db_type)
    if not db_status['status']:
        raise Exception('Unable to access database')
        
    if db_service.db_type == 'RAGatouille':
        raise ValueError(f"Unsupported database type: {db_service.db_type}. RAGatouille not supported.")

    # Get documents and questions dataframes
    docs_df = _get_dataframe_from_store(
        db_service.db_type,
        db_service.index_name
    )
    docs_df["type"] = "doc"

    return docs_df

def add_clusters(df, n_clusters, llm_service=None, docs_per_cluster: int = 10):
    """
    Add cluster labels to DataFrame using KMeans clustering.
    """
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(np.stack(df['embedding'].values))
    
    # Generate cluster labels using LLM if provided
    if llm_service is not None:
        cluster_labels = []
        for i in range(n_clusters):
            cluster_docs = df[df['cluster'] == i]['page_content'].head(docs_per_cluster).tolist()
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
    """
    Export DataFrame to Hugging Face dataset.
    """
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(
        dataset_name, 
        token=os.getenv('HUGGINGFACEHUB_API_KEY')
    )

def get_available_indexes(db_type, embedding_model=None):
    """
    Get available indexes based on current settings. If embedding_model is None, 
    returns all indexes without filtering on those criteria.
    """
    logger = logging.getLogger(__name__)

    def _process_pinecone_indexes():
        """
        Process available Pinecone indexes.
        """
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        
        available_indexes = []
        index_metadatas = []
        
        for index_name in db_status['message']:                
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
        """
        Process available RAGatouille indexes.
        """
        available_indexes = []
        index_metadatas = []
        
        for index_name in db_status['message']:
            available_indexes.append(index_name)
            # RAGatouille doesn't support metadata storage currently
            index_metadatas.append({})
            
        return available_indexes, index_metadatas

    # Get database status
    db_status = get_database_status(db_type)
    logger.info(f"Database status in get_available_indexes for {db_type}, {embedding_model}: {db_status}")
    
    if not db_status['status']:
        return [], []
        
    try:
        if db_type == 'Pinecone':
            available_indexes, index_metadatas = _process_pinecone_indexes()
        elif db_type == 'RAGatouille':
            available_indexes, index_metadatas = _process_ragatouille_indexes()
            
        return available_indexes, index_metadatas
        
    except Exception as e:
        logger.error(f"Error in get_available_indexes: {str(e)}")
        return [], []

def get_database_status(db_type):
    """
    Get status of database indexes/collections.
    """
    def _get_pinecone_status():
        """Get status of Pinecone indexes."""
        
        try:
            pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            indexes = pc.list_indexes()
            if not indexes:
                return {'status': False, 'message': 'No indexes found'}
            return {'status': True, 'message': [idx.name for idx in indexes]}
        except Exception as e:
            return {'status': False, 'message': f'Error connecting to Pinecone: {str(e)}'}

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

    if db_type == 'Pinecone':
        return _get_pinecone_status()
    elif db_type == 'RAGatouille':
        return _get_ragatouille_status()
    else:
        return {'status': False, 'message': f'Unsupported database type: {db_type}'}