"""Document processing and chunking logic."""

import os, hashlib, json
from typing import List, Optional, Any
from dataclasses import dataclass
from langchain_core.documents import Document
from google.cloud import storage
import streamlit as st
import tempfile

from ..core.cache import Dependencies
from ..services.prompts import SUMMARIZE_TEXT

@dataclass
class ChunkingResult:
    """Container for chunking results."""
    rag_type: str
    pages: List[Any]
    chunks: Optional[List[Any]] = None
    splitters: Optional[Any] = None
    n_merge_pages: Optional[int] = None
    chunk_method: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    parent_chunks: Optional[List[Any]] = None
    summaries: Optional[List[Document]] = None
    llm_service: Optional[Any] = None

class DocumentProcessor:
    """Handles document processing, chunking, and indexing."""
    
    def __init__(self, 
                 db_service,
                 embedding_service,
                 rag_type='Standard',
                 chunk_method='RecursiveCharacterTextSplitter',
                 chunk_size=500,
                 chunk_overlap=0,
                 merge_pages=None,
                 llm_service=None):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.rag_type = rag_type
        self.chunk_method = chunk_method
        self.splitter = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.merge_pages = merge_pages
        self.llm_service = llm_service  # Only for rag_type=='Summary', otherwise ignored
        self._deps = Dependencies()

    def process_documents(self, documents, show_progress=False):
        """Process documents based on RAG type by chunking."""
        self.show_progress = show_progress

        cleaned_docs = self._load_and_clean_documents(documents)
        
        if self.rag_type == 'Standard':
            return self._process_standard(cleaned_docs)
        elif self.rag_type == 'Parent-Child':
            return self._process_parent_child(cleaned_docs)
        elif self.rag_type == 'Summary':
            return self._process_summary(cleaned_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {self.rag_type}")

    def _load_and_clean_documents(self, documents):
        """Load PDF documents and clean their contents."""
        # TODO use cache load function
        from langchain_community.document_loaders import PyPDFLoader
        
        if self.show_progress:
            progress_text = 'Reading documents...'
            my_bar = st.progress(0, text=progress_text)
        
        cleaned_docs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, doc in enumerate(documents):
                if self.show_progress:
                    progress_percentage = i / len(documents)
                    my_bar.progress(progress_percentage, text=f'Reading documents...{doc}...{progress_percentage*100:.2f}%')

                # Handle GCS URLs
                if doc.startswith('gs://'):
                    # Parse bucket and blob name
                    bucket_name = doc.split('/')[2]
                    blob_name = '/'.join(doc.split('/')[3:])
                    
                    # Download file from GCS
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    local_path = f"{temp_dir}/{blob_name.split('/')[-1]}"
                    blob.download_to_filename(local_path)
                    doc = local_path

                # Load and process the PDF
                loader = PyPDFLoader(doc)
                doc_page_data = loader.load()

                # Clean up page info, update some metadata
                doc_pages = []
                for doc_page in doc_page_data:
                    doc_page = self._sanitize_page(doc_page)
                    if doc_page is not None:
                        doc_pages.append(doc_page)

                # Merge pages if option is selected
                if self.merge_pages is not None:
                    for i in range(0, len(doc_pages), self.merge_pages):
                        group = doc_pages[i:i+self.merge_pages]
                        group_page_content = ' '.join([doc.page_content for doc in group])
                        group_metadata = {'page': str([doc.metadata['page'] for doc in group]), 
                                        'source': str([doc.metadata['source'] for doc in group])}
                        merged_doc = Document(page_content=group_page_content, metadata=group_metadata)
                        cleaned_docs.append(merged_doc)
                else:
                    cleaned_docs.extend(doc_pages)
        
        if self.show_progress:
            my_bar.empty()
        
        return cleaned_docs

    def index_documents(self,
                       chunking_result,
                       index_name,
                       batch_size=100,
                       clear=False):
        """Index processed documents. This is where the index is initialized or created if required."""
        OpenAIEmbeddings, VoyageAIEmbeddings, HuggingFaceInferenceAPIEmbeddings = self._deps.get_embedding_deps()

        if self.show_progress:  
            progress_text = "Document indexing in progress..."
            my_bar = st.progress(0, text=progress_text)

        self.vectorstore = self.db_service.initialize_database(
            index_name=index_name,
            embedding_service=self.embedding_service,
            rag_type=self.rag_type,
            namespace=self.db_service.namespace,
            clear=clear
        )

        # Add index metadata
        index_metadata = {}
        if chunking_result.n_merge_pages is not None:
            index_metadata['n_merge_pages'] = chunking_result.n_merge_pages
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

        # Index chunks in batches
        for i in range(0, len(chunking_result.chunks), batch_size):
            batch = chunking_result.chunks[i:i + batch_size]
            batch_ids = [self._hash_metadata(chunk.metadata) for chunk in batch]
            
            if self.db_service.db_type == "Pinecone":
                self.vectorstore.add_documents(documents=batch, ids=batch_ids, namespace=self.db_service.namespace)
            else:
                self.vectorstore.add_documents(documents=batch, ids=batch_ids)

            if self.show_progress:
                progress_percentage = min(1.0, (i + batch_size) / len(chunking_result.chunks))
                my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')

        # Handle parent documents or summaries if needed
        if self.rag_type in ['Parent-Child', 'Summary']:
            self._store_parent_docs(index_name, chunking_result, self.rag_type)

        if self.show_progress:
            my_bar.empty()
    @staticmethod
    def list_available_buckets():
        """Lists all available buckets in the GCS project."""
        try:
            # Initialize the GCS client
            storage_client = storage.Client()
            
            # List all buckets
            buckets = [bucket.name for bucket in storage_client.list_buckets()]
            
            return buckets
        except Exception as e:
            raise Exception(f"Error accessing GCS buckets: {str(e)}")
    @staticmethod
    def list_bucket_pdfs(bucket_name: str):
        """Lists all PDF files in a Google Cloud Storage bucket."""
        try:
            # Initialize the GCS client
            storage_client = storage.Client()
            
            # Get the bucket
            bucket = storage_client.bucket(bucket_name)
            
            # List all blobs (files) in the bucket
            blobs = bucket.list_blobs()
            
            # Filter for PDF files and create full GCS paths
            pdf_files = [
                f"gs://{bucket_name}/{blob.name}" 
                for blob in blobs 
                if blob.name.endswith('.pdf')
            ]
            
            return pdf_files
        except Exception as e:
            raise Exception(f"Error accessing GCS bucket: {str(e)}")
    def copy_vectors(self, index_name, source_namespace, target_namespace, batch_size=100, show_progress=False):
        """Copies vectors from a source Pinecone namespace to a target namespace."""
        if self.db_service.db_type != 'Pinecone':
            raise ValueError("Vector copying is only supported for Pinecone databases")
            
        # FIXME initialize database first, this function deosn't exist
        index = self.db_service.get_index(index_name)
        
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
    def _process_standard(self, documents):
        """Process documents for standard RAG."""
        chunks = self._chunk_documents(documents)
        return ChunkingResult(rag_type=self.rag_type,
                              pages=documents,
                              chunks=chunks, 
                              splitters=self.splitter,
                              chunk_method=self.chunk_method,
                              n_merge_pages=self.merge_pages,
                              chunk_size=self.chunk_size,
                              chunk_overlap=self.chunk_overlap)
    def _process_parent_child(self, documents):
        """Process documents for parent-child RAG."""
        chunks, parent_chunks = self._chunk_documents(documents)

        # chunks, {'doc_ids': doc_ids, 'parent_chunks': parent_chunks}
        return ChunkingResult(rag_type=self.rag_type,
                              pages=parent_chunks,
                              chunks=chunks, 
                              splitters={'parent_splitter':self.parent_splitter,'child_splitter':self.child_splitter},
                              n_merge_pages=self.merge_pages,
                              chunk_size=self.chunk_size,
                              chunk_overlap=self.chunk_overlap)
    def _process_summary(self, documents):
        """Process documents for summary RAG."""
        # TODO fix the dependencies, use cache
        from langchain_core.output_parsers import StrOutputParser

        if self.show_progress:
            progress_text = 'Chunking documents...'
            my_bar = st.progress(0, text=progress_text)
        
        chunks = self._chunk_documents(documents)

        # Create unique ids for each chunk, set up chain
        id_key = "doc_id"
        doc_ids = [str(self._stable_hash_meta(chunk.metadata)) for chunk in chunks]
        # Setup the summarization chain
        chain = (
            {"doc": lambda x: x.page_content}
            | SUMMARIZE_TEXT
            | self.llm_service.get_llm()
            | StrOutputParser()
        )
        
        # Process documents in batches
        summaries = []
        batch_size=10   # TODO make this a parameter
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_summaries = chain.batch(batch, config={"max_concurrency": batch_size})
            summaries.extend(batch_summaries)
            if self.show_progress:
                progress_percentage = i / len(chunks)
                my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
        
        # Create summary documents with metadata
        summary_docs = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]        

        if self.show_progress:
            my_bar.empty()  
            
        return ChunkingResult(rag_type=self.rag_type,
                              pages={'doc_ids':doc_ids,'docs':chunks},
                              summaries=summary_docs, 
                              llm_service=self.llm_service,
                              n_merge_pages=self.merge_pages,
                              chunk_method=self.chunk_method,
                              chunk_size=self.chunk_size,
                              chunk_overlap=self.chunk_overlap,
        )

    def _chunk_documents(self, documents):
        """Chunk documents using specified parameters."""
        # TODO import from cache function
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        if self.show_progress:
            progress_text = 'Chunking documents...'
            my_bar = st.progress(0, text=progress_text)

        chunks = []
        if self.rag_type != 'Parent-Child':
            if self.chunk_method == 'None':
                if self.show_progress:
                    my_bar.empty()
                return documents
            for i, doc in enumerate(documents):
                if self.chunk_method == 'character_recursive':
                    self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                                   chunk_overlap=self.chunk_overlap,
                                                                   add_start_index=True)
                    page_chunks = self.splitter.split_documents([doc])
                    chunks.extend(page_chunks)  # Use extend to flatten the list
                else:
                    raise NotImplementedError
                if self.show_progress:
                    progress_percentage = i / len(documents)
                    my_bar.progress(progress_percentage, text=f'Chunking documents...{progress_percentage*100:.2f}%')
            
            if self.show_progress:
                my_bar.empty()
            return chunks
        elif self.rag_type == 'Parent-Child':
            parent_chunks = []
            if self.chunk_method == 'None':
                parent_chunks = documents
                for i, doc in enumerate(documents):
                    self.k_child = 4
                    doc_ids = [str(self._stable_hash_meta(parent_chunk.metadata)) for parent_chunk in parent_chunks]
                    id_key = "doc_id"
                    for parent_chunk in parent_chunks:
                        _id = doc_ids[i]
                        # Default character recursive since no chunking method specified, use k_child to determine chunk size
                        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=len(parent_chunk.page_content) / self.k_child, 
                                                                            chunk_overlap=0,
                                                                            add_start_index=True)
                        _chunks = self.child_splitter.split_documents([parent_chunk])
                        for _doc in _chunks:
                            _doc.metadata[id_key] = _id
                        chunks.extend(_chunks)  # Use extend to flatten the list
                    if self.show_progress:
                        progress_percentage = i / len(documents)
                        my_bar.progress(progress_percentage, text=f'Chunking parent-child documents...{progress_percentage*100:.2f}%')
                if self.show_progress:
                    my_bar.empty()
                return chunks, {'doc_ids': doc_ids, 'parent_chunks': parent_chunks}
            else:
                for i, doc in enumerate(documents):
                    self.k_child = 4
                    if self.chunk_method == 'character_recursive':
                        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                                            chunk_overlap=self.chunk_overlap,
                                                                            add_start_index=True)
                        parent_page_chunks = self.parent_splitter.split_documents([doc])
                        parent_chunks.extend(parent_page_chunks)  # Use extend to flatten the list
                    else:
                        raise NotImplementedError
                    if self.show_progress:
                        progress_percentage = i / len(documents)
                        my_bar.progress(progress_percentage, text=f'Chunking parent documents...{progress_percentage*100:.2f}%')
                    
                doc_ids = [str(self._stable_hash_meta(parent_chunk.metadata)) for parent_chunk in parent_chunks]
                id_key = "doc_id"
                for i, doc in enumerate(parent_chunks):
                    _id = doc_ids[i]
                    if self.chunk_method == 'character_recursive':
                        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size / self.k_child, 
                                                                            chunk_overlap=self.chunk_overlap,
                                                                            add_start_index=True)
                    else:
                        raise NotImplementedError
                    _chunks = self.child_splitter.split_documents([doc])
                    for _doc in _chunks:
                        _doc.metadata[id_key] = _id
                    chunks.extend(_chunks)  # Use extend to flatten the list
                    if self.show_progress:
                        progress_percentage = i / len(documents)
                        my_bar.progress(progress_percentage, text=f'Chunking child documents...{progress_percentage*100:.2f}%')
        
                if self.show_progress:
                    my_bar.empty()
                return chunks, {'doc_ids': doc_ids, 'parent_chunks': parent_chunks}
        else:
            raise NotImplementedError
    @staticmethod
    def _sanitize_page(doc):
        """Clean up page content and metadata."""
        import re
        
        # Clean up content
        content = doc.page_content
        content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", content)
        content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", content.strip())
        content = re.sub(r"\n\s*\n", "\n\n", content)
        
        # Validate content
        if len(content) == 0:
            return None
            
        num_words = len(content.split())
        alphanumeric_pct = sum(c.isalnum() for c in content) / len(content)
        
        if num_words < 5 or alphanumeric_pct < 0.3:
            return None
            
        doc.page_content = content
        return doc

    @staticmethod
    def _hash_metadata(metadata):
        """Create stable hash from metadata."""
        import hashlib
        import json
        return hashlib.sha1(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _merge_pages(docs, n_pages):
        """Merge consecutive pages."""
        merged = []
        for i in range(0, len(docs), n_pages):
            batch = docs[i:i + n_pages]
            merged_content = "\n\n".join(d.page_content for d in batch)
            merged_metadata = batch[0].metadata.copy()
            merged_metadata['merged_pages'] = n_pages
            merged.append(Document(
                page_content=merged_content,
                metadata=merged_metadata
            ))
        return merged

    @staticmethod
    def _upload_to_gcs(bucket_name, file_path, local_file_path):
        """Upload a file to Google Cloud Storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.upload_from_filename(local_file_path)

    def _store_parent_docs(self, index_name, chunking_result, rag_type):
        """Store parent documents or original documents for Parent-Child or Summary RAG types."""
        from pathlib import Path
        import json
        import os

        # Create local file store directory if it doesn't exist
        lfs_path = Path(os.getenv('LOCAL_DB_PATH')).resolve() / 'local_file_store' / index_name
        lfs_path.mkdir(parents=True, exist_ok=True)

        if rag_type == 'Parent-Child':
            # Store parent documents
            for doc_id, parent_doc in zip(chunking_result.metadata['doc_ids'], chunking_result.parent_chunks):
                file_path = lfs_path / str(doc_id)
                with open(file_path, "w") as f:
                    json.dump({"kwargs": {"page_content": parent_doc.page_content}}, f)
        
        elif rag_type == 'Summary':
            # Store original documents
            for doc_id, orig_doc in zip(chunking_result.metadata['doc_ids'], chunking_result.pages['docs']):
                file_path = lfs_path / str(doc_id)
                with open(file_path, "w") as f:
                    json.dump({"kwargs": {"page_content": orig_doc.page_content}}, f)
    @staticmethod
    def _stable_hash_meta(metadata):
        """Calculates the stable hash of the given metadata dictionary."""
        return hashlib.sha1(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    def _store_index_metadata(self, index_metadata):
        """Store index metadata based on the database type."""

        embedding_size = self.embedding_service.get_dimension()
        metadata_vector = [1e-5] * embedding_size

        # index = self.db_service.vectorstore
        if self.db_service.db_type == "Pinecone":
            # TODO use cache dependency function
            # TODO see if I can do this with langchain vectorstore, problem is I need to make a dunmmy embedding.
            from pinecone import Pinecone as pinecone_client
            
            pc_native = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc_native.Index(self.db_service.index_name)
            index.upsert(vectors=[{
                'id': 'db_metadata',
                'values': metadata_vector,
                'metadata': index_metadata
            }])
        
        elif self.db_service.db_type == "Chroma":
            # TODO use cache dependency function
            from chromadb import PersistentClient
            chroma_native = PersistentClient(path=os.path.join(os.getenv('LOCAL_DB_PATH'),'chromadb'))    
            index = chroma_native.get_collection(name=self.db_service.index_name)
            index.add(
                embeddings=[metadata_vector],
                metadatas=[index_metadata],
                ids=['db_metadata']
            )

        
        elif self.db_service.db_type == "Ragatsouille":
            # TODO add metadata storage for RAGatouille, maybe can use the same method as others
            print("Warning: Metadata storage is not yet supported for RAGatouille indexes")