"""UI components for the aerospace chatbot."""

from .admin import SidebarManager
from .utils import (
    setup_page_config,
    display_chat_history, 
    display_sources, 
    show_connection_status,
    extract_pages_from_pdf,
    get_pdf,
    get_or_create_spotlight_viewer,
)

__all__ = [
    'SidebarManager',
    'setup_page_config',
    'display_chat_history', 
    'display_sources', 
    'show_connection_status',
    'extract_pages_from_pdf',
    'get_pdf',
    'get_or_create_spotlight_viewer',
]
