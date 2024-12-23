"""Document processing and chunking logic."""

import hashlib, json, re, os
from typing import List, Optional, Any
from dataclasses import dataclass
import tempfile
import logging

# TODO: Smarter imports
import unstructured_client
from unstructured_client.models import operations, shared
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_dicts

from ..core.cache import Dependencies, cache_data
from ..services.prompts import SUMMARIZE_TEXT

@dataclass
class ChunkingResult:
    """Container for chunking results."""
    rag_type: str
    chunks: List[Any]
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    parent_chunks: List[Any] = None
    summary_chunks: Optional[List[Any]] = None
    llm_service: Optional[Any] = None

class DocumentProcessor:
    """Handles document processing, chunking, and indexing."""
    
    def __init__(
        self, 
        embedding_service,
        work_dir='./unstructured_results',
        rag_type='Standard',
                #  chunk_method='character_recursive',
        chunk_size=500,
        chunk_overlap=0,
        #  merge_pages=None,
        llm_service=None
    ):
        self.embedding_service = embedding_service
        self.work_dir = work_dir
        self.rag_type = rag_type
        # self.chunk_method = chunk_method
        self.splitter = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # self.merge_pages = merge_pages
        self.llm_service = llm_service  # Only for rag_type=='Summary', otherwise ignored
        self._deps = Dependencies()
        self.logger = logging.getLogger(__name__)

        os.makedirs(self.work_dir, exist_ok=True)

        if self.rag_type == 'Summary' and not self.llm_service:
            raise ValueError("LLM service is required for Summary RAG type")

    def load_and_partition_documents(
        self,
        documents,
        partition_by_api=True,
        upload_bucket=None
    ):
        """
        Load and partition documents using either the Unstructured API or local processing.
        partition_by_api specifies if unstructured API is to be used or local.
        If upload_bucket is specified, upload the partitioned documents to the specified bucket on GCS.
        """

        def partition_with_api(pdf_path):
            client = unstructured_client.UnstructuredClient(
                api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
                server_url=os.getenv("UNSTRUCTURED_API_URL"),
            )

            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=open(pdf_path, "rb"),
                        file_name=pdf_path,
                    ),
                    strategy=shared.Strategy.HI_RES,
                    languages=['eng'],
                    coordinates=True,
                    split_pdf_page=True,            # If True, splits the PDF file into smaller chunks of pages.
                    split_pdf_allow_failed=True,    # If True, the partitioning continues even if some pages fail.
                    split_pdf_concurrency_level=15  # Set the number of concurrent request to the maximum value (number of pages): 15.
                ),
            )
            res = client.general.partition(request=req)
            return [element for element in res.elements]
        
        def partition_locally(pdf_path):
            elements = partition_pdf(
                pdf_path,
                strategy="hi_res",
                infer_table_structure=True
            )
            return [element.to_dict() for element in elements]

        self.logger.info(f"Loading {len(documents)} documents...")
        docs_to_process = self._load_documents(documents)

        self.logger.info(f"Partitioning {len(docs_to_process)} documents...")
        output_dir=os.path.join(self.work_dir, 'partitioned')
        os.makedirs(output_dir, exist_ok=True)

        # Process directory of PDFs
        partitioned_docs = []
        for pdf_file in docs_to_process:
            if partition_by_api:
                self.logger.info(f"Partitioning {pdf_file} with Unstructured API...")
                partitioned_data = partition_with_api(pdf_file)
            else:
                self.logger.info(f"Partitioning {pdf_file} Locally...")
                partitioned_data = partition_locally(pdf_file)

            # Save the partitioned data to JSON
            output_path = os.path.join(
                output_dir, 
                os.path.basename(os.path.splitext(pdf_file)[0]) + "-partitioned.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(partitioned_data, f, ensure_ascii=False, indent=4)

            self.logger.info(f"Partitioned data saved at {output_path}")
            partitioned_docs.append(output_path)
            if upload_bucket:
                self._upload_to_gcs(upload_bucket, pdf_file) # Upload original PDF file to GCS
                self._upload_to_gcs(upload_bucket, output_path) # Upload JSON file to GCS
                self.logger.info(f"Uploaded {pdf_file} and {output_path} to {upload_bucket}")

        return partitioned_docs
    
    def process_documents(self, partitioned_docs):
        """
        Chunk documents based on RAG type.
        Partitioned docs is either a list of local or GCS json files.
        If no_chunk is True, skip the chunking step and only partition documents.
        """

        # Load partitioned docs
        # If local, do nothing. If GCS, download to work_dir and update partitioned_docs to local paths

        self.logger.info("Chunking documents...")
        if self.rag_type == 'Standard':
            return self._chunk_standard(partitioned_docs)
        elif self.rag_type == 'Parent-Child':
            return self._chunk_parent_child(partitioned_docs)
        elif self.rag_type == 'Summary':
            return self._chunk_summary(partitioned_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {self.rag_type}")
            
    @staticmethod
    def list_available_buckets():
        """Lists all available buckets in the GCS project."""
        _, _, storage, _, _, _ = Dependencies.Document.get_processors()

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
        _, _, storage, _, _, _ = Dependencies.Document.get_processors()

        logger = logging.getLogger(__name__)
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
            logger.info(f"Number of PDFs found: {len(pdf_files)}")
            logger.info(f"PDFs found: {pdf_files}")
            return pdf_files
        except Exception as e:
            raise Exception(f"Error accessing GCS bucket: {str(e)}")
    
    @staticmethod
    def stable_hash_meta(metadata):
        """Calculates the stable hash of the given metadata dictionary."""
        return hashlib.sha1(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    
    def _load_documents(self, documents):
        """
        Load PDF documents and return a list of local file paths. Accepts GCS URLs and local file paths.
        
        work_dir is the directory to store downloaded and processed PDF files.
        """
        _, _, storage, PyPDFLoader, _, _ = Dependencies.Document.get_processors()

        def _download_and_validate_pdf(doc_in):
            """Download and validate a PDF document."""
            
            # Handle GCS URLs
            if doc_in.startswith('gs://'):
                self.logger.info(f"Downloading PDF from GCS: {doc_in}")
                bucket_name = doc_in.split('/')[2]
                blob_name = '/'.join(doc_in.split('/')[3:])
                self.logger.info(f"Bucket name: {bucket_name}")
                self.logger.info(f"Blob name: {blob_name}")
                
                # Download file from GCS
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                local_path = f"{self.work_dir}/{blob_name.split('/')[-1]}"
                blob.download_to_filename(local_path)
                doc_local_temp = local_path
            else:
                doc_local_temp = doc_in

            # Validate the PDF
            try:
                loader = PyPDFLoader(doc_local_temp)
                loader.load()
            except Exception as e:
                self.logger.error(f"Invalid PDF document: {doc_in}. Error: {str(e)}")
                return None

            return doc_local_temp
        
        docs_to_process = []
        invalid_docs = []
        for i, doc_in in enumerate(documents):
            self.logger.info(f"Checking document {i+1} of {len(documents)}: {doc_in}")

            doc_local_temp = _download_and_validate_pdf(doc_in)
            if doc_local_temp is None:
                invalid_docs.append(doc_in)
                continue
            else:
                docs_to_process.append(doc_local_temp)

        if invalid_docs:
            raise ValueError(f"Invalid PDF documents detected: {', '.join(invalid_docs)}")
        
        return docs_to_process
    
    def _chunk_standard(self, documents):
        """Chunk documents for standard RAG."""
        chunks = self._chunk_documents(documents)
        self.logger.info(f"Number of chunks: {len(chunks)}")
        return ChunkingResult(rag_type=self.rag_type,
                              chunks=chunks,
                              chunk_size=self.chunk_size,
                              chunk_overlap=self.chunk_overlap)
    def _chunk_parent_child(self, documents):
        """Chunk documents for parent-child RAG."""
        chunks, parent_chunks = self._chunk_documents(documents)
        self.logger.info(f"Number of chunks: {len(chunks)}")
        return ChunkingResult(rag_type=self.rag_type,
                              chunks=chunks,
                              parent_chunks=parent_chunks,
                              chunk_size=self.chunk_size,
                              chunk_overlap=self.chunk_overlap)
    def _chunk_summary(self, documents):
        """Chunk documents for summary RAG."""
        _, StrOutputParser, _, _, _, _, Document, _ = Dependencies.LLM.get_chain_utils()
        
        chunks = self._chunk_documents(documents)

        # Create unique ids for each chunk, set up chain
        id_key = "doc_id"
        doc_ids = [str(self.stable_hash_meta(chunk.metadata)) for chunk in chunks]
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
        
        # Create summary documents with metadata
        summary_chunks = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]        
            
        self.logger.info(f"Number of summaries: {len(summary_chunks)}")
        return ChunkingResult(rag_type=self.rag_type,
                              chunks={'doc_ids':doc_ids,'chunks':chunks},
                              summary_chunks=summary_chunks, 
                              llm_service=self.llm_service,
                              chunk_size=self.chunk_size,
                              chunk_overlap=self.chunk_overlap,
        )

    def _chunk_documents(self, documents):
        """Chunk documents using specified parameters."""
        RecursiveCharacterTextSplitter = Dependencies.Document.get_splitters()

        # TODO Create chunking functionality here for unstructured
        # Provide load from work_dir if a partitioned directory exists with json files
        # Also provide the option to load from GCS if partitioned files are present in GCS

        # Check for partitioned input as a bucket or local directory

        chunks = []
        if self.rag_type != 'Parent-Child':
            # if self.chunk_method == 'None' or self.chunk_method is None:
            #     return documents
            for i, doc in enumerate(documents):
                # if self.chunk_method == 'character_recursive':
                self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                                chunk_overlap=self.chunk_overlap,
                                                                add_start_index=True)
                page_chunks = self.splitter.split_documents([doc])
                chunks.extend(page_chunks)  # Use extend to flatten the list
                # else:
                #     raise NotImplementedError
            return chunks
        elif self.rag_type == 'Parent-Child':
            parent_chunks = []
            # if self.chunk_method == 'None':
            #     parent_chunks = documents
            #     for i, doc in enumerate(documents):
            #         self.k_child = 4
            #         doc_ids = [str(self.stable_hash_meta(parent_chunk.metadata)) for parent_chunk in parent_chunks]
            #         id_key = "doc_id"
            #         for parent_chunk in parent_chunks:
            #             _id = doc_ids[i]
            #             # Default character recursive since no chunking method specified, use k_child to determine chunk size
            #             self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=len(parent_chunk.page_content) / self.k_child, 
            #                                                                 chunk_overlap=0,
            #                                                                 add_start_index=True)
            #             _chunks = self.child_splitter.split_documents([parent_chunk])
            #             for _doc in _chunks:
            #                 _doc.metadata[id_key] = _id
            #             chunks.extend(_chunks)  # Use extend to flatten the list
            #     return chunks, {'doc_ids': doc_ids, 'parent_chunks': parent_chunks}
            # else:
            for i, doc in enumerate(documents):
                self.k_child = 4
                # if self.chunk_method == 'character_recursive':
                self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                                    chunk_overlap=self.chunk_overlap,
                                                                    add_start_index=True)
                parent_page_chunks = self.parent_splitter.split_documents([doc])
                parent_chunks.extend(parent_page_chunks)  # Use extend to flatten the list
                # else:
                #     raise NotImplementedError
                
            doc_ids = [str(self.stable_hash_meta(parent_chunk.metadata)) for parent_chunk in parent_chunks]
            id_key = "doc_id"
            for i, doc in enumerate(parent_chunks):
                _id = doc_ids[i]
                # if self.chunk_method == 'character_recursive':
                self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size / self.k_child, 
                                                                    chunk_overlap=self.chunk_overlap,
                                                                    add_start_index=True)
                # else:
                #     raise NotImplementedError
                _chunks = self.child_splitter.split_documents([doc])
                for _doc in _chunks:
                    _doc.metadata[id_key] = _id
                chunks.extend(_chunks)  # Use extend to flatten the list
            return chunks, {'doc_ids': doc_ids, 'parent_chunks': parent_chunks}
        else:
            raise NotImplementedError
    @staticmethod
    def _sanitize_page(doc):
        """Clean up page content and metadata."""
        
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
    def _merge_pages(docs, n_pages):
        """Merge consecutive pages."""
        _, _, _, _, _, _, Document, _ = Dependencies.LLM.get_chain_utils()

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
    def _upload_to_gcs(bucket_name, file_path):
        """Upload a file to Google Cloud Storage."""
        _, _, storage, _, _, _ = Dependencies.Document.get_processors()

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.upload_from_filename(file_path)

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