"""Document processing and chunking logic."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_core.documents import Document

from ..core.cache import Dependencies

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

    def _process_summary(self, documents):
        """Process documents for summary RAG."""
        from ..services.llm import LLMService
        
        chunks = self._chunk_documents(documents)
        summaries = []
        
        # Create summaries using LLM
        llm_service = LLMService(
            model_name="gpt-3.5-turbo",
            model_type="openai",
            api_key=self.embedding_service.api_key
        )
        
        for doc in documents:
            summary = self._generate_summary(doc, llm_service)
            summaries.append(summary)
            
        return ChunkingResult(
            chunks=chunks,
            summaries=summaries,
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
