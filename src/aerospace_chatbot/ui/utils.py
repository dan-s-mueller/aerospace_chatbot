"""UI utility functions."""

import streamlit as st
from typing import List, Dict
import os

from ..core.cache import Dependencies

def get_cache_data_decorator():
    """Returns appropriate cache_data decorator based on environment"""
    try:
        import streamlit as st
        return st.cache_data
    except:
        # Return no-op decorator when not in Streamlit
        return lambda *args, **kwargs: (lambda func: func)

# Replace @st.cache_data with dynamic decorator
cache_data = get_cache_data_decorator()

def setup_page_config(title: str = "Aerospace Chatbot", layout: str = "wide"):
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=title,
        layout=layout,
        page_icon="🚀"
    )

def display_chat_history(history: List[Dict], show_metadata: bool = False):
    """Display chat history with optional metadata."""
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if show_metadata and "metadata" in msg:
                with st.expander("Message Metadata"):
                    st.json(msg["metadata"])

def display_sources(sources: List[Dict], expanded: bool = False):
    """Display reference sources in an expander."""
    with st.expander("Sources", expanded=expanded):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}:**")
            st.markdown(f"- **Title:** {source.get('title', 'N/A')}")
            st.markdown(f"- **Page:** {source.get('page', 'N/A')}")
            if source.get('url'):
                st.markdown(f"- **URL:** [{source['url']}]({source['url']})")
            st.markdown("---")

@cache_data
def show_connection_status(expanded: bool = True):
    """Display connection status for various services."""
    with st.expander("Connection Status", expanded=expanded):
        # API Keys Status
        st.markdown("**API Keys Status:**")
        _display_api_key_status()
        
        # Database Status
        st.markdown("**Database Status:**")
        _display_database_status()
        
        # Model Status
        st.markdown("**Model Status:**")
        _display_model_status()

def _display_api_key_status():
    """Display API key status."""
    keys = {
        'OpenAI API Key': os.getenv('OPENAI_API_KEY'),
        'Anthropic API Key': os.getenv('ANTHROPIC_API_KEY'),
        'Voyage API Key': os.getenv('VOYAGE_API_KEY'),
        'Pinecone API Key': os.getenv('PINECONE_API_KEY'),
        'Hugging Face API Key': os.getenv('HUGGINGFACEHUB_API_TOKEN')
    }
    
    for name, key in keys.items():
        status = "✅" if key else "❌"
        st.markdown(f"- {name}: {status}")

@cache_data
def _display_database_status():
    """Display database connection status."""
    from ..services.database import DatabaseService
    
    db_service = DatabaseService()
    status = db_service.check_connection()
    st.markdown(f"- Database Connection: {'✅' if status else '❌'}")

@cache_data
def _display_model_status():
    """Display model availability status."""
    from ..services.llm import LLMService
    
    llm_service = LLMService()
    status = llm_service.check_availability()
    st.markdown(f"- Model Availability: {'✅' if status else '❌'}")

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
