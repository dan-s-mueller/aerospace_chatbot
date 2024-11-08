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
    # TODO add handling for missing keys. First check if the key is needed based on the config file. Make the list of required secrets. Then check if they are in the environment. If not, raise an error.
    load_dotenv(find_dotenv())
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'HUGGINGFACEHUB_API_KEY': os.getenv('HUGGINGFACEHUB_API_KEY'),
        'VOYAGE_API_KEY': os.getenv('VOYAGE_API_KEY'),
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY')
    }

def set_secrets(secrets):
    """Sets environment variables from provided secrets dictionary."""
    for key_name, value in secrets.items():
        if value:  # Only set if value exists
            os.environ[key_name] = value
            if os.environ[key_name] == '':
                readable_name = ' '.join(key_name.split('_')[:-1]).title()
                raise ConfigurationError(f'{readable_name} is required.')
    
    return secrets
