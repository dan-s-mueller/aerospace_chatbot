"""Configuration management and environment setup."""

import os, json
from dotenv import load_dotenv, find_dotenv

class ConfigurationError(Exception):
    """Raised when there are configuration related errors."""
    pass
def get_cache_decorator():
    """Returns appropriate cache decorator based on environment."""
    try:
        import streamlit as st
        return st.cache_resource
    except ImportError:
        # Return no-op decorator when not in Streamlit
        return lambda *args, **kwargs: (lambda func: func)

def get_cache_data_decorator():
    """Returns appropriate cache_data decorator based on environment."""
    try:
        import streamlit as st
        return st.cache_data
    except ImportError:
        # Return no-op decorator when not in Streamlit
        return lambda *args, **kwargs: (lambda func: func)
@get_cache_data_decorator()
def load_config(config_path):
    """Load configuration from file with caching. """
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path) as f:
            config = json.load(f)
            
        # Validate config structure
        required_sections = ['databases', 'embeddings', 'llms', 'rag_types']
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")
                
        return config
        
    except Exception as e:
        raise ConfigurationError(f"Failed to load config: {str(e)}")
def get_secrets():
    """Load and return secrets from environment"""
    load_dotenv(find_dotenv())
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'HUGGINGFACEHUB_API_TOKEN': os.getenv('HUGGINGFACEHUB_API_TOKEN'),
        'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY'),
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY')
    }
