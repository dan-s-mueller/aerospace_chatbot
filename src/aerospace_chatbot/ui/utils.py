"""UI utility functions."""

import streamlit as st
import os

from ..core.cache import Dependencies, get_cache_data_decorator
from ..services.database import DatabaseService

cache_data = get_cache_data_decorator()

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

def display_sources(sources, expanded = False):
    """Display reference sources in an expander."""
    # TODO add back old pdf display functionality
    with st.expander("Sources", expanded=expanded):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}:**")
            st.markdown(f"- **Title:** {source.get('title', 'N/A')}")
            st.markdown(f"- **Page:** {source.get('page', 'N/A')}")
            if source.get('url'):
                st.markdown(f"- **URL:** [{source['url']}]({source['url']})")
            st.markdown("---")

def show_connection_status(expanded = True, delete_buttons = False):
    """Display connection status for various services with optional delete functionality. """
    with st.expander("Connection Status", expanded=expanded):
        # API Keys Status
        st.markdown("**API Keys Status:**")
        _display_api_key_status()
        
        # Database Status and Management
        st.markdown("**Database Status:**")
        _display_database_status(delete_buttons)
@cache_data
def extract_pages_from_pdf(url, target_page, page_range=5):
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
@cache_data
def get_pdf(url):
    """Downloads complete PDF file."""
    deps = Dependencies()
    fitz, requests = deps.get_pdf_deps()
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_data = response.content

        doc = fitz.open("pdf", pdf_data)
        extracted_pdf = doc.tobytes()
        doc.close()
        return extracted_pdf

    except Exception as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None
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
    keys = {
        'OpenAI API Key': os.getenv('OPENAI_API_KEY'),
        'Anthropic API Key': os.getenv('ANTHROPIC_API_KEY'),
        'Voyage API Key': os.getenv('VOYAGE_API_KEY'),
        'Pinecone API Key': os.getenv('PINECONE_API_KEY'),
        'Hugging Face API Key': os.getenv('HUGGINGFACEHUB_API_TOKEN')
    }
    
    markdown_str = "\n".join([f"- {name}: {'✅' if key else '❌'}" for name, key in keys.items()])
    st.markdown(markdown_str)

def _display_database_status(delete_buttons = False):
    """Display database status and management options."""
    # Get local_db_path from environment
    local_db_path = os.getenv('LOCAL_DB_PATH')
    if not local_db_path:
        st.error("Local database path not set")
        return

    # Initialize database services for each type
    db_services = {
        'Pinecone': DatabaseService('Pinecone', local_db_path),
        'ChromaDB': DatabaseService('ChromaDB', local_db_path),
        'RAGatouille': DatabaseService('RAGatouille', local_db_path)
    }
    
    # Show Pinecone status
    pinecone_status = db_services['Pinecone'].get_pinecone_status()
    st.markdown("**Pinecone Indexes:**")
    if pinecone_status['status']:
        for index in pinecone_status['indexes']:
            st.markdown(f"- `{index}` ✅")
            if delete_buttons:
                _handle_index_deletion('Pinecone', index, db_services['Pinecone'])
    else:
        st.markdown(f"- {pinecone_status['message']} ❌")

    # Show ChromaDB status
    chroma_status = db_services['ChromaDB'].get_chroma_status()
    st.markdown("**ChromaDB Collections:**")
    if chroma_status['status']:
        for collection in chroma_status['collections']:
            st.markdown(f"- `{collection.name}` ✅")
            if delete_buttons:
                _handle_index_deletion('ChromaDB', collection.name, db_services['ChromaDB'])
    else:
        st.markdown(f"- {chroma_status['message']} ❌")

    # Show RAGatouille status
    rag_status = db_services['RAGatouille'].get_ragatouille_status()
    st.markdown("**RAGatouille Indexes:**")
    if rag_status['status']:
        for index in rag_status['indexes']:
            st.markdown(f"- `{index}` ✅")
            if delete_buttons:
                _handle_index_deletion('RAGatouille', index, db_services['RAGatouille'])
    else:
        st.markdown(f"- {rag_status['message']} ❌")

    # Show database path
    st.markdown(f"**Local database path:** `{local_db_path}`")

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