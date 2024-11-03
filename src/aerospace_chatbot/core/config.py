"""Configuration management and environment setup."""

import os
import json
from pathlib import Path

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

class ConfigurationError(Exception):
    """Raised when there are configuration related errors."""
    pass

@get_cache_data_decorator()
def load_config(config_path = None):
    """Load configuration from file with caching. """
    if config_path is None:
        config_path = os.getenv('AEROSPACE_CHATBOT_CONFIG_PATH')
        if not config_path:
            raise ConfigurationError("No config path specified")
    
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

def ensure_paths():
    """Ensure required application paths exist."""
    required_paths = {
        'LOCAL_DB_PATH': os.getenv('LOCAL_DB_PATH'),
        'CACHE_DIR': os.getenv('CACHE_DIR', 'cache'),
    }
    
    for name, path in required_paths.items():
        if not path:
            raise ConfigurationError(f"Missing required path: {name}")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
