"""UI utility functions."""

import streamlit as st
import os, ast, tempfile
from streamlit_pdf_viewer import pdf_viewer

from ..core.cache import Dependencies
from ..core.config import get_secrets
from ..services.database import DatabaseService
from ..services.embeddings import EmbeddingService
from ..processing.documents import DocumentProcessor

def setup_page_config(title: str = "Aerospace Chatbot", layout: str = "wide"):
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=title,
        layout=layout,
        page_icon="🚀"
    )
def display_chat_history(history, show_metadata = False):
    """Display chat history with optional metadata."""
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if show_metadata and "metadata" in msg:
                with st.expander("Message Metadata"):
                    st.json(msg["metadata"])
def display_sources(sources, expanded=False):
    """Display reference sources in an expander with PDF preview functionality."""
    with st.container():
        with st.spinner('Bringing you source documents...'):
            st.write(":notebook: Source Documents")
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
                    selected_url = f"https://storage.googleapis.com/aerospace_mechanisms_chatbot_demo/{pdf_source}"
                    st.markdown(f"[{pdf_source} (Download)]({selected_url}) - Page {page}")

                    with st.expander(":memo: View"):
                        tab1, tab2 = st.tabs(["Relevant Context+5 Pages", "Full"])
                        try:
                            # Extract and display the pages when the user clicks
                            extracted_pdf = _extract_pages_from_pdf(selected_url, page)
                            with tab1:
                                pdf_viewer(extracted_pdf, width=1000, height=1200, render_text=True)
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
        # FIXME the delete buttons don't work
        # TODO make one todo button with a select box to delete it
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
            _process_uploads(sb, temp_files)

    if st.session_state.user_upload:
        st.markdown(
            f":white_check_mark: Merged! Your upload ID: `{st.session_state.user_upload}`. "
            "This will be used for this chat session to also include your documents. "
            "When you restart the chat, you'll have to re-upload your documents."
        )
def get_or_create_spotlight_viewer(df, port: int = 9000):
    """Create or get existing Spotlight viewer instance.
    
    Args:
        df: pandas DataFrame containing the data to visualize
        port: port number to run the viewer on (default: 9000)
        
    Returns:
        Spotlight viewer instance
    """
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

    db_services = {
        'Pinecone': DatabaseService('Pinecone', os.getenv('LOCAL_DB_PATH')),
        'ChromaDB': DatabaseService('ChromaDB', os.getenv('LOCAL_DB_PATH')),
        'RAGatouille': DatabaseService('RAGatouille', os.getenv('LOCAL_DB_PATH'))
    }
    
    # Display status for each database type
    for db_type, service in db_services.items():
        st.markdown(f"**{db_type} {'Indexes' if db_type != 'ChromaDB' else 'Collections'}:**")
        status = service.get_database_status(db_type)
        
        if status['status']:
            for index in status['indexes']:
                # Handle different index name formats
                index_name = index.name if hasattr(index, 'name') else index
                st.markdown(f"- `{index_name}` ✅")
                if delete_buttons:
                    _handle_index_deletion(db_type, index_name, service)
        else:
            st.markdown(f"- {status['message']} ❌")

    # Show database path
    st.markdown(f"**Local database path:** `{os.getenv('LOCAL_DB_PATH')}`")
def _handle_index_deletion(db_type, index_name, db_service):
    """Handle deletion of database indexes."""
    if st.button(f'Delete {index_name}', help='This is permanent!'):
        try:
            rag_type = _determine_rag_type(index_name)
            db_service.delete_index(db_type, index_name, rag_type)
            st.success(f"Successfully deleted {index_name}")
            st.rerun()  # Refresh the page to show updated status
        except Exception as e:
            st.error(f"Error deleting index: {str(e)}")
def _determine_rag_type(index_name):
    """Determine RAG type from index name.
    
    Args:
        index_name: Name of the index
        
    Returns:
        RAG type ('Parent-Child', 'Summary', or 'Standard')
    """
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
    
    st.write("Upload parameters set to standard values, hard coded for now...standard only")
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
def _process_uploads(sb, temp_files):
    """Process uploaded files and merge them into the vector database."""
    # Initialize services
    db_service = DatabaseService(
        db_type=sb['index_type']
    )
    embedding_service = EmbeddingService(
        model_name=sb['embedding_name'],
        model_type=sb['query_model']
    )
    # Initialize document processor
    doc_processor = DocumentProcessor(
        db_service=db_service,
        embedding_service=embedding_service
    )
    
    # Generate unique identifier
    st.session_state.user_upload = f"user_upload_{os.urandom(3).hex()}"
    st.markdown("*Processing and uploading user documents...*")
    
    # Process and index documents
    st.markdown("*Uploading user documents to namespace...*")
    chunking_result = doc_processor.process_documents(temp_files)
    doc_processor.index_data(
        index_name=sb['index_selected'],
        data=chunking_result,
        namespace=st.session_state.user_upload,
        show_progress=True
    )
    # Copy vectors to merge namespaces
    st.markdown("*Merging user document with existing documents...*")
    doc_processor.copy_vectors(
        index_name=sb['index_selected'],
        source_namespace=None,
        target_namespace=st.session_state.user_upload,
        show_progress=True
    )
# @cache_data
def _extract_pages_from_pdf(url, target_page, page_range=5):
    """Extracts specified pages from a PDF file."""
    deps = Dependencies()
    fitz, requests = deps.get_pdf_deps()
    
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