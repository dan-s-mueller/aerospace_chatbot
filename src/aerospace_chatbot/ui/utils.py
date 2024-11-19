"""UI utility functions."""

import streamlit as st
import os, ast, tempfile, logging
from streamlit_pdf_viewer import pdf_viewer

from ..core.cache import Dependencies
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
        previous_state.get('rag_type') != current_state.get('rag_type') or
        previous_state.get('embedding_model') != current_state.get('embedding_model') or
        previous_state.get('embedding_service') != current_state.get('embedding_service') or
        previous_state.get('llm_service') != current_state.get('llm_service')):
        st.rerun()
    
    return current_state

def display_sources(sources, expanded=False):
    """Display reference sources in an expander with PDF preview functionality."""
    with st.container():
        with st.spinner('Bringing you source documents...'):
            for source in sources:
                page = source.get('page')
                pdf_source = source.get('source')
                
                # Parse string representations of lists
                try:
                    page = ast.literal_eval(page) if isinstance(page, str) else page
                    pdf_source = ast.literal_eval(pdf_source) if isinstance(pdf_source, str) else pdf_source
                except (ValueError, SyntaxError):
                    pass

                # Extract first element if it's a list
                page = page[0] if isinstance(page, list) and page else page
                pdf_source = pdf_source[0] if isinstance(pdf_source, list) and pdf_source else pdf_source
                
                if pdf_source and page is not None:
                    selected_url = f"https://storage.googleapis.com/{pdf_source}"
                    st.markdown(f"[{pdf_source} (Download)]({selected_url}) - Page {page}")

                    with st.expander(":memo: View"):
                        tab1, tab2 = st.tabs(["Relevant Context+5 Pages", "Full"])
                        try:
                            # Extract and display the pages when the user clicks
                            extracted_pdf = _extract_pages_from_pdf(selected_url, page)
                            with tab1:
                                pdf_viewer(extracted_pdf, width=600, height=1200, render_text=True)
                            with tab2:
                                st.write("Disabled for now...see download link above!")
                        except Exception as e:
                            st.warning("Unable to load PDF preview. Either the file no longer exists or is inaccessible. Contact support if this issue persists. User file uploads not yet supported.")
def show_connection_status(expanded = True, delete_buttons = False):
    """Display connection status for various services with optional delete functionality. """
    with st.expander("Connection Status", expanded=expanded):
        # API Keys Status
        st.markdown("**API Keys Status:**")
        _display_api_key_status()
        
        # Database Status and Management
        st.markdown("**Database Status:**")
        _display_database_status(delete_buttons)
def handle_file_upload(sb):
    """Handle file upload functionality for the chatbot."""
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
    """Process uploaded files and merge them into the vector database."""
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
        embedding_model=sb['embedding_model'],
        rag_type=sb['rag_type']
    )

    logger.info(f"Available indexes: {available_indexes}")
    logger.info(f"Index metadatas: {index_metadatas}")
    logger.info(f"Selected index: {sb['index_selected']}")

    if sb['index_selected'] not in available_indexes:
        raise ValueError(f"Selected index {sb.get('index_selected')} not found for compatible index type {sb.get('index_type')}, rag type {sb.get('rag_type')}, and embedding model {sb.get('embedding_model')}")
    else:
        logger.info(f"Selected index {sb['index_selected']} found for compatible index type {sb['index_type']}, rag type {sb['rag_type']}, and embedding model {sb['embedding_model']}")

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
        rag_type=sb['rag_type'],
        embedding_service=embedding_service,
        doc_type='document'
    )
    db_service.initialize_database(namespace=user_upload)
    logger.info(f"Initialized database with namespace {db_service.namespace}")

    # Initialize document processor with default values if metadata fields don't exist
    doc_processor = DocumentProcessor(
        embedding_service=embedding_service,
        rag_type=sb['rag_type'],
        chunk_method=selected_metadata.get('chunk_method', None),
        chunk_size=selected_metadata.get('chunk_size', None),
        chunk_overlap=selected_metadata.get('chunk_overlap', None),
        merge_pages=selected_metadata.get('merge_pages', None)
    )
    
    # Process and index documents
    logger.info("Uploading user documents to namespace...")
    chunking_result = doc_processor.process_documents(temp_files)
    db_service.index_data(
        data=chunking_result
    )
    # Copy vectors to merge namespaces
    logger.info("Merging user document with existing documents...")
    db_service.copy_vectors(
        source_namespace=None
    )
    logger.info("Merged user document with existing documents.")
    return user_upload

def get_or_create_spotlight_viewer(df, port: int = 9000):
    """Create or get existing Spotlight viewer instance."""
    deps = Dependencies()
    spotlight = deps.get_spotlight()
    
    viewer = spotlight.show(
        df,
        port=port,
        return_viewer=True,
        open_browser=False
    )
    return viewer

def _display_api_key_status():
    """Display API key status."""
    secrets = get_secrets()
    markdown_str = "\n".join([f"- {name}: {'✅' if secret else '❌'}" for name, secret in secrets.items()])
    st.markdown(markdown_str)

def _display_database_status(delete_buttons=False):
    """Display database status and management options."""
    if not os.getenv('LOCAL_DB_PATH'):
        st.error("Local database path not set")
        return

    db_types = ['Pinecone', 'ChromaDB', 'RAGatouille']
    rag_types = ['Standard', 'Parent-Child', 'Summary']
    
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
    """Handle deletion of database indexes."""
    if st.button(f'Delete {index_name}', help='This is permanent!'):
        try:
            rag_type = _determine_rag_type(index_name)
            if index_name.endswith('-queries'):
                doc_type = 'question'
            else:
                doc_type = 'document'
            db_service = DatabaseService(
                db_type=db_type,
                index_name=index_name,
                rag_type=rag_type,
                embedding_service=None,
                doc_type=doc_type
            )
            db_service.delete_index()
            st.success(f"Successfully deleted {index_name}")
            st.rerun()  # Refresh the page to show updated status
        except Exception as e:
            st.error(f"Error deleting index: {str(e)}")

def _determine_rag_type(index_name):
    """Determine RAG type from index name."""
    if index_name.endswith('-parent-child'):
        return 'Parent-Child'
    elif '-summary-' in index_name or index_name.endswith('-summary'):
        return 'Summary'
    return 'Standard'

def _validate_upload_settings(sb):
    """Validate RAG type and index type settings."""
    if sb['rag_type'] != "Standard":
        st.error("Only Standard RAG is supported for user document upload.")
        return False
    if sb['index_type'] != 'Pinecone':
        st.error("Only Pinecone is supported for user document upload.")
        return False
    
    st.write("Upload parameters determined by index selected in sidebar.")
    return True

def _save_uploads_to_temp(uploaded_files):
    """Save uploaded files to temporary directory."""
    temp_files = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())
        temp_files.append(temp_path)
    return temp_files

def _extract_pages_from_pdf(url, target_page, page_range=5):
    """Extracts specified pages from a PDF file."""
    fitz, requests, _, _ = Dependencies.Document.get_processors()
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_data = response.content

        doc = fitz.open("pdf", pdf_data)
        extracted_doc = fitz.open()

        start_page = max(target_page, 0)
        end_page = min(target_page + page_range, doc.page_count - 1)

        for i in range(start_page, end_page + 1):
            extracted_doc.insert_pdf(doc, from_page=i, to_page=i)

        extracted_pdf = extracted_doc.tobytes()
        extracted_doc.close()
        doc.close()
        return extracted_pdf

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None