"""Document processing and chunking logic."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from google.cloud import storage

from ..core.cache import Dependencies
from ..services.prompts import SUMMARIZE_TEXT

@dataclass
class ChunkingResult:
    """Container for chunking results."""
    chunks: List[Document]
    metadata: Dict[str, Any]
    parent_chunks: Optional[List[Document]] = None
    summaries: Optional[List[Document]] = None

class DocumentProcessor:
    """Handles document processing, chunking, and indexing."""
    
    def __init__(self, 
                 db_service,
                 embedding_service,
                 chunk_size=500,
                 chunk_overlap=50):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._deps = Dependencies()

    def process_documents(self, 
                         documents,
                         rag_type='Standard',
                         merge_pages=1):
        """Process documents based on RAG type."""
        # Sanitize and merge pages if needed
        cleaned_docs = [self._sanitize_page(doc) for doc in documents]
        cleaned_docs = [doc for doc in cleaned_docs if doc is not None]
        
        if merge_pages > 1:
            cleaned_docs = self._merge_pages(cleaned_docs, merge_pages)

        if rag_type == 'Standard':
            return self._process_standard(cleaned_docs)
        elif rag_type == 'Parent-Child':
            return self._process_parent_child(cleaned_docs)
        elif rag_type == 'Summary':
            return self._process_summary(cleaned_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")

    def index_documents(self,
                       index_name,
                       chunking_result,
                       rag_type='Standard',
                       batch_size=100,
                       namespace=None):
        """Index processed documents."""
        vectorstore = self.db_service.initialize_database(
            index_name=index_name,
            embedding_service=self.embedding_service,
            rag_type=rag_type,
            namespace=namespace,
            clear=True
        )

        # Index chunks in batches
        for i in range(0, len(chunking_result.chunks), batch_size):
            batch = chunking_result.chunks[i:i + batch_size]
            batch_ids = [self._hash_metadata(chunk.metadata) for chunk in batch]
            
            if self.db_service.db_type == "pinecone":
                vectorstore.add_documents(documents=batch, ids=batch_ids, namespace=namespace)
            else:
                vectorstore.add_documents(documents=batch, ids=batch_ids)

        # Handle parent documents or summaries if needed
        if rag_type in ['Parent-Child', 'Summary']:
            self._store_parent_docs(index_name, chunking_result, rag_type)

    def _process_standard(self, documents):
        """Process documents for standard RAG."""
        chunks = self._chunk_documents(documents)
        return ChunkingResult(chunks=chunks, metadata={'rag_type': 'Standard'})

    def _process_parent_child(self, documents):
        """Process documents for parent-child RAG."""
        chunks = self._chunk_documents(documents)
        parent_chunks = documents
        
        # Add parent document references to chunks
        for chunk in chunks:
            chunk.metadata['doc_id'] = self._hash_metadata(chunk.metadata)
            
        return ChunkingResult(
            chunks=chunks,
            parent_chunks=parent_chunks,
            metadata={'rag_type': 'Parent-Child'}
        )

    def _process_summary(self, documents, llm, batch_size=10, show_progress=False):
        """Process documents for summary RAG."""
        from ..services.llm import LLMService
        from langchain_core.output_parsers import StrOutputParser
        
        chunks = self._chunk_documents(documents)
        
        # Setup the summarization chain
        chain = (
            {"doc": lambda x: x.page_content}
            | SUMMARIZE_TEXT
            | llm
            | StrOutputParser()
        )
        
        # Generate document IDs
        doc_ids = [self._hash_metadata(doc.metadata) for doc in documents]
        
        # Initialize progress bar if requested
        if show_progress:
            try:
                import streamlit as st
                progress_bar = st.progress(0, text='Generating summaries...')
            except ImportError:
                show_progress = False
        
        # Process documents in batches
        summaries = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_summaries = chain.batch(batch, config={"max_concurrency": batch_size})
            summaries.extend(batch_summaries)
            
            # Update progress if enabled
            if show_progress:
                progress = min(1.0, (i + batch_size) / len(documents))
                progress_bar.progress(progress, text=f'Generating summaries...{progress*100:.2f}%')
        
        # Create summary documents with metadata
        summary_docs = [
            Document(page_content=summary, metadata={"doc_id": doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        
        # Clean up progress bar
        if show_progress:
            progress_bar.empty()
        
        return ChunkingResult(
            chunks=chunks,
            summaries=summary_docs,
            metadata={'rag_type': 'Summary'}
        )

    def _chunk_documents(self, documents):
        """Chunk documents using specified parameters."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            
        return chunks

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
                if blob.name.lower().endswith('.pdf')
            ]
            
            return pdf_files
        except Exception as e:
            raise Exception(f"Error accessing GCS bucket: {str(e)}")
