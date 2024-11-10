"""Database service implementations."""

import os, shutil
import re
from pathlib import Path
import time
import json

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
        self.retriever = None
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
            self._init_pinecone(clear=clear)
        elif self.db_type == 'ChromaDB':
            self._init_chromadb(clear=clear)
        elif self.db_type == 'RAGatouille':
            self._init_ragatouille(clear=clear)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def index_documents(self,
                       chunking_result,
                       batch_size=100,
                       clear=False):
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
        self._upsert_docs(chunking_result, batch_size, clear=clear)

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
    
    def copy_vectors(self, source_namespace, target_namespace, batch_size=100):
        # FIXME, moved over without testing from documents
        """Copies vectors from a source Pinecone namespace to a target namespace."""
        if self.db_type != 'Pinecone':
            raise ValueError("Vector copying is only supported for Pinecone databases")
            
        # FIXME initialize database first, this function deosn't exist
        index = self.get_index(self.index_name)
        
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
            pinecone_client, _, _ = self._deps.get_db_deps()
   
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

    def _init_chromadb(self, clear=False):
        """Initialize ChromaDB."""
        _, chromadb, _ = self._deps.get_db_deps()
        _, Chroma, _, _ = self._deps.get_vectorstore_deps()
        
        db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        
        if clear:
            self.delete_index()

        self.vectorstore = Chroma(
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
        from ragatouille import RAGPretrainedModel
        
        index_path = os.path.join(os.getenv('LOCAL_DB_PATH'), '.ragatouille')
        if clear:
            self.delete_index()
        self.vectorstore = RAGPretrainedModel.from_pretrained(
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

    def _upsert_docs(self, chunking_result, batch_size, clear=False):
        """Upsert documents. Used for all RAG types. Clear only used for RAGatouille."""
        # TODO update to use dependency cache
        from langchain.storage import LocalFileStore
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from ..processing.documents import DocumentProcessor
        
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
                
                # Index parent docs all at once
                retriever.docstore.mset(list(zip(doc_ids, pages)))
                self.retriever = retriever

def get_docs_questions_df(db_service, query_db_service):
    """Get documents and questions from database as a DataFrame."""
    deps = Dependencies()
    pd, _, _, _ = deps.get_analysis_deps()
    
    def _fetch_pinecone_docs(index):
        """Fetch documents from Pinecone in batches."""
        ids = []
        for id_batch in index.list():
            ids.extend(id_batch)
            
        docs = []
        chunk_size = 200
        for i in range(0, len(ids), chunk_size):
            vector = index.fetch(ids[i:i+chunk_size])['vectors']
            vector_data = [value for value in vector.values()]
            docs.extend(vector_data)
        return docs

    def _process_chromadb_response(response, doc_type):
        """Process ChromaDB response into DataFrame."""
        if doc_type == 'document':
            return pd.DataFrame({
                "id": response["ids"],
                "source": [metadata.get("source") for metadata in response["metadatas"]],
                "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
                "metadata": response["metadatas"],
                "document": response["documents"],
                "embedding": response["embeddings"],
            })
        else:  # question
            return pd.DataFrame({
                "id": response["ids"],
                "question": response["documents"],
                "answer": [metadata.get("answer") for metadata in response["metadatas"]],
                "sources": [metadata.get("sources", "").split(",") for metadata in response["metadatas"]],
                "embedding": response["embeddings"],
            })

    def _process_pinecone_response(docs, doc_type):
        """Process Pinecone response into DataFrame."""
        if doc_type == 'document':
            return pd.DataFrame({
                "id": [data['id'] for data in docs],
                "source": [data['metadata']['source'] for data in docs],
                "page": [data['metadata']['page'] for data in docs],
                "metadata": [{'page':data['metadata']['page'],'source':data['metadata']['source']} for data in docs],
                "document": [data['metadata']['page_content'] for data in docs],
                "embedding": [data['values'] for data in docs],
            })
        else:  # question
            return pd.DataFrame({
                "id": [data['id'] for data in docs],
                "question": [data['metadata']['page_content'] for data in docs],
                "answer": [data['metadata'].get('answer', '') for data in docs],
                "sources": [data['metadata'].get('sources', '').split(',') for data in docs],
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

    def _get_dataframe_from_store(vectorstore, db_type, index_name, doc_type='document'):
        """Common method to retrieve data from any vectorstore type and convert to DataFrame."""
        try:
            if db_type == 'ChromaDB':
                response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
                df = _process_chromadb_response(response, doc_type)
                
            elif db_type == 'Pinecone':
                docs = _fetch_pinecone_docs(vectorstore.index)
                df = _process_pinecone_response(docs, doc_type)
                
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Add RAG-specific columns if needed
            if doc_type == 'document':
                df = _add_rag_columns(df, index_name)
                
            return df
            
        except Exception as e:
            print(f"Error getting {doc_type}s: {str(e)}")
            columns = ['id', 'source', 'page', 'metadata', 'document', 'embedding'] if doc_type == 'document' else \
                     ['id', 'question', 'answer', 'sources', 'embedding']
            return pd.DataFrame(columns=columns)

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
    
    print(f"Query database found: {query_db_service.index_name}")

    # Get documents and questions dataframes
    docs_df = _get_dataframe_from_store(
        db_service.vectorstore,
        db_service.db_type,
        db_service.index_name,
        doc_type='document'
    )
    docs_df["type"] = "doc"
    print(f"Retrieved docs from: {db_service.index_name}")

    questions_df = _get_dataframe_from_store(
        query_db_service.vectorstore,
        query_db_service.db_type,
        query_db_service.index_name,
        doc_type='question'
    )
    questions_df["type"] = "question"
    print(f"Retrieved questions from: {query_db_service.index_name}")

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
    """Add cluster labels to DataFrame using KMeans clustering. Adding LLM service will generate cluster labels."""
    deps = Dependencies()
    _, np, KMeans, _ = deps.get_analysis_deps()
    
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

def export_to_hf_dataset(df, dataset_name):
    """Export DataFrame to Hugging Face dataset."""
    deps = Dependencies()
    _, _, _, Dataset = deps.get_analysis_deps()
    
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(
        dataset_name, 
        token=os.getenv('HUGGINGFACEHUB_API_KEY')
    )

def get_available_indexes(db_type, embedding_model, rag_type):
    """Get available indexes based on current settings."""
    # Get database status
    db_status = get_database_status(db_type)
    
    if not db_status['status']:
        return [], []
        
    available_indexes = []
    index_metadatas = []
    
    def _check_get_index_criteria(index_name):
        """Check if index meets RAG type criteria."""
        if index_name.endswith('-queries'):
            return False
            
        return (rag_type == 'Parent-Child' and index_name.endswith('-parent-child')) or \
                (rag_type == 'Summary' and index_name.endswith('-summary')) or \
                (rag_type == 'Standard' and not index_name.endswith(('-parent-child', '-summary')))

    def _process_chromadb_indexes():
        """Process available ChromaDB indexes."""
        _, chromadb, _ = Dependencies().get_db_deps()
        client = chromadb.PersistentClient(path=os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb'))
        
        for index in db_status['message']:
            if not _check_get_index_criteria(index.name):
                continue
            collection = client.get_collection(index.name)
            metadata = collection.get(ids=['db_metadata'])
            if metadata and metadata['metadatas'][0].get('embedding_model') == embedding_model:
                available_indexes.append(index.name)
                index_metadatas.append(metadata['metadatas'][0])

    def _process_pinecone_indexes():
        """Process available Pinecone indexes."""
        from pinecone import Pinecone
        pc = Pinecone()
        
        for index_name in db_status['message']:
            if not _check_get_index_criteria(index_name):
                continue
            index_obj = pc.Index(index_name)
            metadata = index_obj.fetch(ids=['db_metadata'])
            if metadata and metadata['vectors']['db_metadata']['metadata'].get('embedding_model') == embedding_model:
                available_indexes.append(index_name)
                index_metadatas.append(metadata['vectors']['db_metadata']['metadata'])

    def _process_ragatouille_indexes():
        """Process available RAGatouille indexes."""
        for index_name in db_status['message']:
            if _check_get_index_criteria(index_name):
                available_indexes.append(index_name)
                # RAGatouille doesn't support metadata storage currently
                index_metadatas.append({})

    try:
        if db_type == 'ChromaDB':
            _process_chromadb_indexes()
        elif db_type == 'Pinecone':
            _process_pinecone_indexes()
        elif db_type == 'RAGatouille':
            _process_ragatouille_indexes()
    except Exception:
        return [], []

    return available_indexes, index_metadatas

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
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        indexes = pc.list_indexes()
        if not indexes:
            return {'status': False, 'message': 'No indexes found'}
        return {'status': True, 'message': [idx.name for idx in indexes]}
    except Exception as e:
        return {'status': False, 'message': f'Error connecting to Pinecone: {str(e)}'}

def _get_chromadb_status():
    """Get status of ChromaDB collections."""
    try:
        import chromadb
        db_path = os.path.join(os.getenv('LOCAL_DB_PATH'), 'chromadb')
        client = chromadb.PersistentClient(path=str(db_path))
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