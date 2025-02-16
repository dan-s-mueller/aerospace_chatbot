"""UI utility functions."""



# Utilities
import streamlit as st
import os, tempfile, logging, re
import fitz
import requests
from PIL import Image
import io
import json, base64, zlib
from typing import List, Dict, Any

# from ..core.cache import Dependencies
from ..core.config import get_secrets
from ..services.database import DatabaseService, get_available_indexes
from ..services.embeddings import EmbeddingService
from ..processing.documents import DocumentProcessor

def setup_page_config(title: str = "Aerospace Chatbot", layout: str = "wide"):
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=title,
        layout=layout,
        page_icon="🚀"
    )

def handle_sidebar_state(sidebar_manager):
    """Handle sidebar state management and dependencies."""
    # Get previous state for comparison
    previous_state = st.session_state.sb.copy() if st.session_state.sb else {}
    
    # Render sidebar
    current_state = sidebar_manager.render_sidebar()
    
    # Check if critical values changed that would affect dependencies
    if (previous_state.get('index_type') != current_state.get('index_type') or
        previous_state.get('embedding_model') != current_state.get('embedding_model') or
        previous_state.get('embedding_service') != current_state.get('embedding_service') or
        previous_state.get('llm_service') != current_state.get('llm_service')):
        st.rerun()
    
    return current_state

def replace_source_tags(content, sources, message_id):
    """
    Replace source tags with hyperlinks in the content.
    """
    def replace_tag(match):
        # Convert 1-indexed to 0-indexed
        source_id = int(match.group(1)) - 1
        if 0 <= source_id < len(sources):
            source = sources[source_id]
            pdf_source = source[0].metadata['data_source.url'].replace('gs://', '')
            selected_url = f"https://storage.googleapis.com/{pdf_source}"
            page_number = int(source[0].metadata['page_number'])
            return f'(<a href="{selected_url}#page={page_number}" target="_blank">📝 Source {message_id}.{source_id + 1}: {os.path.basename(pdf_source)}, page: {page_number}</a>)'
        return match.group(0)  # Return the original tag if no source found

    return re.sub(r'<source id="(\d+)">', replace_tag, content)

def display_sources(sources, n_display, message_id, expanded=False):
    """
    Display reference sources in an expander with PDF preview functionality.
    """
    logger = logging.getLogger(__name__)

    annotated_pdfs, page_ranges, pdf_sources, selected_urls = process_source_documents(sources, n_display)
    
    with st.container():
        with st.spinner('Bringing you source documents...'):
            for i, (annotated_pdf, page_range, pdf_source, selected_url) in enumerate(zip(annotated_pdfs, page_ranges, pdf_sources, selected_urls)):
                # Style the expander
                st.markdown("""
                    <style>
                        .stExpander {
                            max-height: 1000px;
                            overflow-y: auto;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                # Display PDF content
                score=sources[i][2]
                formatted_score = f"{score:.3f}"    # Format the score to 3 decimal places
                with st.expander(f":memo: Source {message_id}.{i+1} (Score: {formatted_score})", expanded=expanded):
                    selected_url = f"https://storage.googleapis.com/{pdf_source}"
                    encoded_url = requests.utils.quote(selected_url, safe=':/#')
                    display_name = os.path.basename(pdf_source)  # Get just the filename for display
                    st.markdown(f"[{display_name} (Download)]({encoded_url}) - Page {page_range[0]}")
                    try:
                        display_pdf(annotated_pdf)
                            
                    except Exception as e:
                        logger.error(f"Failed to display source: {e}")
                        st.error("Unable to display PDF. Please use the download link.")
                        raise e

def process_source_documents(sources, n_display):
    """
    Process source documents and display them with highlights.
    Will display the first n_display sources. Recommended that this aligns with the k_rerank docs that are returned.
    """
    logger = logging.getLogger(__name__)

    def extract_pages_from_pdf(url, page_range, page_buffer=3):
        """
        Extracts specified pages from a PDF file.
        page_range is a list with the first and last page number of the source
        page_buffer is the number of pages to include before and after the source
        """     
        if page_buffer<1:
            raise ValueError("Page buffer must be at least 1")
        
        try:
            # Download PDF
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Open PDF and extract pages
            doc = fitz.open(stream=response.content, filetype="pdf")
            original_page_count = doc.page_count
            extracted_doc = fitz.open()
            
            # Add page buffer to the start and end page range, this is now zero indexed
            if page_range[0] == 1:
                # Don't buffer a first page
                start_page = 0
            else:
                start_page = page_range[0] - page_buffer - 1

            if page_range[1] == original_page_count:
                # Don't buffer a last page
                end_page = original_page_count - 1
            else:
                end_page = page_range[1] + page_buffer - 1
            
            extracted_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
            
            # Get bytes and cleanup
            extracted_pdf_bytes = extracted_doc.tobytes()
            extracted_doc.close()
            doc.close()

            logger.info(f"Extracted pdf from {url} for pages (1 indexed) {start_page+1} to {end_page+1}")
            
            return (io.BytesIO(extracted_pdf_bytes), start_page, end_page)
                    
        except Exception as e:
            logger.error(f"Failed to extract PDF pages: {e}")
            raise ValueError(f"PDF extraction failed: {str(e)}")

    def extract_orig_elements(orig_element_metadata):
        """Extract the contents of an orig_elements field."""
        decoded_orig_elements = base64.b64decode(orig_element_metadata)
        decompressed_orig_elements = zlib.decompress(decoded_orig_elements)
        return decompressed_orig_elements.decode('utf-8')
    
    def get_chunked_elements(source):
        # result['context'][0][0].metadata['orig_elements']
        # sources = result['context']
        if "orig_elements" in source[0].metadata:
            orig_elements = extract_orig_elements(source[0].metadata['orig_elements'])
            orig_elements_dict = {
                "element_id": source[0].metadata['element_id'],
                "text": source[0].page_content,
                "orig_elements": json.loads(orig_elements)
            }
        else:
            raise ValueError("No orig_elements field found in document metadata, unable to display sources.")
        return orig_elements_dict
    
    def annotate_pdf_with_highlights(pdf_file, orig_elements, page_range):
        """
        Annotate a PDF with highlights.
        page_range is a list with the first and last page number of the extracted document from the original source
        """
        # Load the PDF
        pdf_bytes = pdf_file.read()
        annotated_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Get PDF dimensions
        first_page = annotated_doc[0]
        pdf_width = first_page.rect.width
        pdf_height = first_page.rect.height

        # Process each chunk
        for chunk in orig_elements['orig_elements']:            
            # Subtract page_range[0] to get the page number in the extracted document with buffer pages, index by 0
            extracted_doc_page_num = chunk['metadata']['page_number'] - page_range[0] - 1
            extracted_doc_page = annotated_doc[extracted_doc_page_num]
            
            # Get pixel dimensions from the element's metadata
            pixel_width = chunk['metadata']['coordinates']['layout_width']
            pixel_height = chunk['metadata']['coordinates']['layout_height']
            
            # Calculate scale factors for both dimensions
            scale_x = pdf_width / pixel_width
            scale_y = pdf_height / pixel_height
            
            points = chunk['metadata']['coordinates']['points']
            
            # Scale the coordinates
            x0 = points[0][0] * scale_x
            y0 = points[0][1] * scale_y
            x2 = points[2][0] * scale_x
            y2 = points[2][1] * scale_y
            
            # Add rectangle annotation
            rect = extracted_doc_page.add_rect_annot(fitz.Rect(x0, y0, x2, y2))
            rect.set_colors(stroke=(1, 1, 0), fill=(1, 1, 0))  # set both outline and fill color. Use yellow for all
            rect.set_opacity(0.3)  # make it semi-transparent
            rect.set_border(width=0.5)  # set border width
            rect.update()

        logger.info(f"Annotated pdf with {len(orig_elements['orig_elements'])} chunks")

        # Get bytes and cleanup
        annotated_doc_bytes = annotated_doc.tobytes()
        annotated_doc.close()

        return io.BytesIO(annotated_doc_bytes)

    annotated_pdfs=[]
    page_ranges=[]
    pdf_sources=[]
    selected_urls=[]
    for source in sources[:n_display]:
        orig_elements = get_chunked_elements(source)

        # pages is the first and last page number of the source
        page_range = [orig_elements['orig_elements'][0]['metadata']['page_number'], orig_elements['orig_elements'][-1]['metadata']['page_number']]
        pdf_source = orig_elements['orig_elements'][0]['metadata']['data_source']['url'].replace('gs://', '')
        
        if pdf_source and page_range:
            selected_url = f"https://storage.googleapis.com/{pdf_source}"
            extracted_pdf, min_page, max_page = extract_pages_from_pdf(selected_url, page_range)

            annotated_pdfs.append(annotate_pdf_with_highlights(extracted_pdf, orig_elements, [min_page, max_page]))
            page_ranges.append(page_range)  # This is 1 indexed since it comes directly from the metadata
            pdf_sources.append(pdf_source)
            selected_urls.append(selected_url)
        else:
            raise ValueError(f"Missing metadata for {source}, unable to display sources.")

    return annotated_pdfs, page_ranges, pdf_sources, selected_urls
    
def display_pdf(annotated_pdf):
    """
    Display a PDF file by converting pages to images.
    """
    logger = logging.getLogger(__name__)

    # Set rendering parameters
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)

    try:
        # Read the PDF
        pdf_bytes = annotated_pdf.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Display each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to bytes for display
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True)
            img_byte_arr = img_byte_arr.getvalue()

            # Display the image
            st.image(
                img_byte_arr,
                use_container_width=True
            )
        pdf_document.close()
    except Exception as e:
        logger.error(f"Failed to display PDF: {e}")
        st.error(f"Error displaying PDF: {e}")

def show_connection_status(expanded = True, delete_buttons = False):
    """
    Display connection status for various services with optional delete functionality. 
    """
    with st.expander("Connection Status", expanded=expanded):
        # API Keys Status
        st.markdown("**API Keys Status:**")
        _display_api_key_status()
        
        # Database Status and Management
        st.markdown("**Database Status:**")
        _display_database_status(delete_buttons)

def handle_file_upload(sb):
    """
    Handle file upload functionality for the chatbot.
    """
    if not _validate_upload_settings(sb):
        return

    uploaded_files = st.file_uploader("Choose pdf files", accept_multiple_files=True)
    if not uploaded_files:
        return

    temp_files = _save_uploads_to_temp(uploaded_files)
    
    if st.button('Upload your docs into vector database'):
        with st.spinner('Uploading and merging your documents with the database...'):
            user_upload = process_uploads(sb, temp_files)
            return user_upload

def process_uploads(sb, temp_files):
    """
    Process uploaded files and merge them into the vector database.
    """
    logger = logging.getLogger(__name__)
    
    # Generate unique identifier
    user_upload=f"user_upload_{os.urandom(3).hex()}"
    logger.info(f"User upload ID: {user_upload}")
    logger.info("Processing and uploading user documents...")

    embedding_service = EmbeddingService(
        model_service=sb['embedding_service'],
        model=sb['embedding_model']
    )

    # Get available indexes and their metadata using the standalone function
    available_indexes, index_metadatas = get_available_indexes(
        db_type=sb['index_type'],
        embedding_model=sb['embedding_model']
    )

    logger.info(f"Available indexes: {available_indexes}")
    logger.info(f"Index metadatas: {index_metadatas}")
    logger.info(f"Selected index: {sb['index_selected']}")

    if sb['index_selected'] not in available_indexes:
        raise ValueError(f"Selected index {sb.get('index_selected')} not found for compatible index type {sb.get('index_type')}, and embedding model {sb.get('embedding_model')}")
    else:
        logger.info(f"Selected index {sb['index_selected']} found for compatible index type {sb['index_type']}, and embedding model {sb['embedding_model']}")

    # Get metadata for the selected index
    selected_metadata = index_metadatas[available_indexes.index(sb['index_selected'])]
    # Convert any float values to int
    for key, value in selected_metadata.items():
        if isinstance(value, float):
            selected_metadata[key] = int(value)
    
    if not selected_metadata:
        raise ValueError(f"No metadata found for index {sb['index_selected']}")
    else:
        logger.info(f"Metadata found for index {sb['index_selected']}")

    # Initialize services
    db_service = DatabaseService(
        db_type=sb['index_type'],
        index_name=sb['index_selected'],
        embedding_service=embedding_service,
    )
    db_service.initialize_database(namespace=user_upload)
    logger.info(f"Initialized database with namespace {db_service.namespace}")

    # Initialize document processor with default values if metadata fields don't exist
    doc_processor = DocumentProcessor(
        embedding_service=embedding_service,
        chunk_size=selected_metadata.get('chunk_size', None),
        chunk_overlap=selected_metadata.get('chunk_overlap', None)
    )
    
    # Process and index documents
    logger.info("Uploading user documents to namespace...")
    partitioned_docs = doc_processor.load_and_partition_documents(
        temp_files,
        partition_by_api=False
    )
    chunk_obj, _ = doc_processor.chunk_documents(partitioned_docs)
    db_service.index_data(chunk_obj) 

    # Copy vectors to merge namespaces
    logger.info("Merging user document with existing documents...")
    db_service.copy_vectors(
        source_namespace=None
    )
    logger.info("Merged user document with existing documents.")
    return user_upload

def _display_api_key_status():
    """
    Display API key status.
    """
    secrets = get_secrets()
    markdown_str = "\n".join([f"- {name}: {'✅' if secret else '❌'}" for name, secret in secrets.items()])
    st.markdown(markdown_str)

def _display_database_status(delete_buttons=False):
    """
    Display database status and management options.
    """
    if not os.getenv('LOCAL_DB_PATH'):
        st.error("Local database path not set")
        return

    db_types = ['Pinecone', 'RAGatouille']
    
    # Display status for each database type
    for db_type in db_types:
        st.markdown(f"**{db_type}:**")
        available_indexes, _ = get_available_indexes(db_type)
        
        if available_indexes:
            for index_name in available_indexes:
                st.markdown(f"- `{index_name}` ✅")
                if delete_buttons:
                    _handle_index_deletion(db_type, index_name)
        else:
            st.markdown(f"- No indexes found ❌")

    # Show database path
    st.markdown(f"**Local database path:** `{os.getenv('LOCAL_DB_PATH')}`")

def _handle_index_deletion(db_type, index_name):
    """
    Handle deletion of database indexes.
    """
    if st.button(f'Delete {index_name}', help='This is permanent!'):
        try:
            if index_name.endswith('-q'):
                doc_type = 'question'
            else:
                doc_type = 'document'
            db_service = DatabaseService(
                db_type=db_type,
                index_name=index_name,
                embedding_service=None,
                doc_type=doc_type
            )
            db_service.delete_index()
            st.success(f"Successfully deleted {index_name}")
            st.rerun()  # Refresh the page to show updated status
        except Exception as e:
            st.error(f"Error deleting index: {str(e)}")

def _validate_upload_settings(sb):
    """
    Validate index type settings.
    """
    if sb['index_type'] != 'Pinecone':
        st.error("Only Pinecone is supported for user document upload.")
        return False
    
    st.write("Upload parameters determined by index selected in sidebar. Upload and merge process may take a while.")
    return True

def _save_uploads_to_temp(uploaded_files):
    """
    Save uploaded files to temporary directory.
    """
    temp_files = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())
        temp_files.append(temp_path)
    return temp_files