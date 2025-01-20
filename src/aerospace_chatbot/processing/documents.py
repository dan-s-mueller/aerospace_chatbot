"""Document processing and chunking logic."""

import hashlib, json, os
import logging

# Parsers
import unstructured_client
from unstructured_client.models import operations, shared
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_dicts
from unstructured.documents.elements import Element

# Utilities
from typing import List, Any, Optional
from google.cloud import storage
from langchain_core.documents import Document
import fitz

# from ..core.cache import Dependencies
# from ..services.prompts import SUMMARIZE_TEXT

class ChunkingResult:
    def __init__(
            self, 
            chunks: List[Element], 
            chunk_size: Optional[int], 
            chunk_overlap: Optional[int]
        ):
        self.chunks = chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_convert(self, destination_type=Document):
        """
        Convert chunks to a destination type.
        """

        def _flatten_metadata(metadata, parent_key='', sep='.'):
            """
            Flatten a nested dictionary and ensure all values are strings, numbers, booleans, or lists of strings.
            """
            items = []
            for k, v in metadata.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_metadata(v, new_key, sep=sep).items())
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    items.append((new_key, v))
                elif isinstance(v, (str, int, float, bool)):
                    items.append((new_key, v))
                else:
                    # Convert unsupported types to string
                    items.append((new_key, str(v)))
            return dict(items)

        converted_chunks = []
        if destination_type == Document:
            for chunk in self.chunks:
                chunk=chunk.to_dict()
                doc_metadata = _flatten_metadata(chunk['metadata'])
                doc_metadata['element_id']=chunk['element_id']
                doc_metadata['type']=chunk['type']
                doc_metadata['chunk_size']=self.chunk_size
                doc_metadata['chunk_overlap']=self.chunk_overlap

                doc=Document(
                    page_content=chunk['text'], 
                    metadata=doc_metadata, 
                )
                converted_chunks.append(doc)
        self.chunks=converted_chunks

class DocumentProcessor:
    """Handles document processing, chunking, and indexing."""
    
    def __init__(
        self, 
        embedding_service,
        work_dir='./document_processing',
        chunk_size=500,
        chunk_overlap=0,
        # merge_pages=None,
        # chunk_method='character_recursive',
        # llm_service=None
    ):
        self.embedding_service = embedding_service
        self.work_dir = work_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

        os.makedirs(self.work_dir, exist_ok=True)

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
            """
            Partition documents using the Unstructured API.
            TODO add in image extraction for multimodal. Not set up with embeddings yet.
            """
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
                    split_pdf_page=True,             # If True, splits the PDF file into smaller chunks of pages.
                    split_pdf_allow_failed=True,     # If True, the partitioning continues even if some pages fail.
                    split_pdf_concurrency_level=15,  # Set the number of concurrent request to the maximum value (number of pages): 15.
                    infer_table_structure=True,      # If True, the partitioning will infer the structure of tables as html.
                    # extract_image_block_types=["Image", "Table"]    # https://docs.unstructured.io/api-reference/how-to/extract-image-block-types#ingest-python-library
                ),
            )
            res = client.general.partition(request=req)
            return [element for element in res.elements]
        
        def partition_locally(pdf_path):
            # TODO check that this is the same functionality as the API partition_pdf function.
            elements = partition_pdf(
                pdf_path,
                strategy="hi_res",
                languages=['eng'],
                split_pdf_page=True,
                split_pdf_allow_failed=True,
                split_pdf_concurrency_level=15,
                infer_table_structure=True,
                # extract_image_block_types=["Image", "Table"],
                # extract_image_block_to_payload=True
            )
            return [element.to_dict() for element in elements]

        self.logger.info(f"Loading {len(documents)} documents...")
        local_docs_to_process, valid_gcs_docs = self._load_documents(documents)

        self.logger.info(f"Partitioning {len(local_docs_to_process)} documents...")
        output_dir=os.path.join(self.work_dir, 'partitioned')
        os.makedirs(output_dir, exist_ok=True)

        # Process directory of PDFs
        partitioned_docs = []
        for pdf_file in local_docs_to_process:
            
            if partition_by_api:
                self.logger.info(f"Partitioning {pdf_file} with Unstructured API...")
                partitioned_data = partition_with_api(pdf_file)
            else:
                self.logger.info(f"Partitioning {pdf_file} Locally...")
                partitioned_data = partition_locally(pdf_file)

            # Add GCS metadata if the PDF has a matching GCS source
            pdf_basename = os.path.basename(pdf_file)
            matching_gcs_doc = next((gcs_doc for gcs_doc in valid_gcs_docs 
                                   if os.path.basename(gcs_doc) == pdf_basename), None)
            if matching_gcs_doc:
                # Parse GCS path components
                bucket_name = matching_gcs_doc.split('/')[2]
                remote_path = f"gs://{bucket_name}"
                
                # Add metadata to each element in partitioned_data
                for element in partitioned_data:
                    element['metadata']['data_source'] = {
                        "url": matching_gcs_doc,
                        "record_locator": {
                            "protocol": "gs",
                            "remote_file_path": remote_path
                        }
                    }

            # Save the partitioned data to JSON
            output_path = os.path.join(
                output_dir, 
                os.path.basename(os.path.splitext(pdf_file)[0]) + "-partitioned.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(partitioned_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Partitioned data saved at {output_path}")
            partitioned_docs.append(output_path)

            # Upload to GCS bucket if specified
            if upload_bucket:
                self._upload_to_gcs(upload_bucket, pdf_file, os.path.basename(pdf_file)) # Upload original PDF file to GCS
                self._upload_to_gcs(upload_bucket, output_path, os.path.join('partitioned', os.path.basename(output_path))) # Upload JSON file to GCS
                self.logger.info(f"Uploaded {pdf_file} and {output_path} to {upload_bucket}")

        return partitioned_docs
    
    def chunk_documents(self, partitioned_docs):
        """
        Chunk documents based on RAG type.
        Partitioned docs is either a list of local or GCS json files.
        """
        # _, _, storage, _, _, _ = Dependencies.Document.get_processors()

        # Process GCS paths if present
        local_partition_paths = []
        for doc_path in partitioned_docs:
            if doc_path.startswith('gs://'):
                # Parse GCS path
                bucket_name = doc_path.split('/')[2]
                blob_name = '/'.join(doc_path.split('/')[3:])
                
                # Download file from GCS
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                # Create local path
                local_path = os.path.join(self.work_dir, os.path.basename(blob_name))
                blob.download_to_filename(local_path)
                local_partition_paths.append(local_path)
                self.logger.info(f"Downloaded {doc_path} to {local_path}")
            else:
                local_partition_paths.append(doc_path)
        
        # Chunk documents
        self.logger.info("Chunking documents...")
        os.makedirs(os.path.join(self.work_dir, 'chunked'), exist_ok=True)
        output_paths = []
        chunks_out = []
        for json_file in local_partition_paths:
            print(f"Chunking {json_file}...")
            
            # Load partitioned data, convert to elements type
            with open(json_file, "r") as file:
                partitioned_data = json.load(file)
            elements=elements_from_dicts(partitioned_data)

            # Chunk the partitioned data by title
            chunks = chunk_by_title(
                elements,
                multipage_sections=True,
                max_characters=self.chunk_size,
                overlap=self.chunk_overlap
            )
            chunks_out.extend(chunks)
            chunks_output = [chunk.to_dict() for chunk in chunks]

            # Save chunked output
            output_path = os.path.join(os.path.join(self.work_dir, 'chunked'), os.path.basename(json_file).replace("-partitioned", "-chunked"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks_output, f, ensure_ascii=False, indent=4)
            output_paths.append(output_path)

            print(f"Chunked data saved at {output_path}")

        self.logger.info(f"Total number of chunks: {len(chunks_out)}")
        self.logger.info(f"Output paths: {output_paths}")

        chunk_obj=ChunkingResult(
            chunks=chunks_out,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        return chunk_obj, output_paths
            
    @staticmethod
    def list_available_buckets():
        """Lists all available buckets in the GCS project."""
        # _, _, storage, _, _, _ = Dependencies.Document.get_processors()

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
        # _, _, storage, _, _, _ = Dependencies.Document.get_processors()

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
        """
        # _, _, storage, PyPDFLoader, _, _ = Dependencies.Document.get_processors()

        def _download_and_validate_pdf(doc_in):
            """Download and validate a PDF document."""
            
            if doc_in.startswith('gs://') and doc_in.lower().endswith('.pdf'):
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
            elif doc_in.lower().endswith('.pdf'):
                doc_local_temp = doc_in
            else:
                self.logger.warning(f"Not a PDF document, skipping: {doc_in}")
                return None

            # Validate the PDF
            try:
                doc = fitz.open(doc_local_temp)
                doc.close()
            except Exception as e:
                self.logger.error(f"Invalid PDF document: {doc_in}. Error: {str(e)}")
                return None

            return doc_local_temp
        
        local_docs_to_process = []
        valid_gcs_docs = []
        invalid_docs = []
        for i, doc_in in enumerate(documents):
            self.logger.info(f"Checking document {i+1} of {len(documents)}: {doc_in}")

            doc_local_temp = _download_and_validate_pdf(doc_in)
            if doc_local_temp is None:
                invalid_docs.append(doc_in)
                continue
            else:
                local_docs_to_process.append(doc_local_temp)
                if doc_in.startswith('gs://'):
                    valid_gcs_docs.append(doc_in)

        if invalid_docs:
            raise ValueError(f"Invalid PDF documents detected: {', '.join(invalid_docs)}")
        
        return local_docs_to_process, valid_gcs_docs
    
    def _chunk_standard(self, documents):
        """Chunk documents for standard RAG."""

        os.makedirs(os.path.join(self.work_dir, 'chunked'), exist_ok=True)
        output_paths = []
        chunks_out = []
        for json_file in documents:
            print(f"Chunking {json_file}...")
            
            # Load partitioned data, convert to elements type
            with open(json_file, "r") as file:
                partitioned_data = json.load(file)
            elements=elements_from_dicts(partitioned_data)

            # Chunk the partitioned data by title
            chunks = chunk_by_title(
                elements,
                multipage_sections=True,
                max_characters=self.chunk_size,
                overlap=self.chunk_overlap
            )
            chunks_out.extend(chunks)
            chunks_output = [chunk.to_dict() for chunk in chunks]

            # Save chunked output
            output_path = os.path.join(os.path.join(self.work_dir, 'chunked'), json_file.replace("-partitioned", "-chunked"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks_output, f, ensure_ascii=False, indent=4)
            output_paths.append(output_path)

            print(f"Chunked data saved at {output_path}")

        self.logger.info(f"Number of chunks: {len(chunks)}")
        return chunks_out, output_paths
    
    @staticmethod
    def _upload_to_gcs(bucket_name, file_local, file_gcs):
        """Upload a file to Google Cloud Storage."""
        # _, _, storage, _, _, _ = Dependencies.Document.get_processors()

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_gcs)
        blob.upload_from_filename(file_local)