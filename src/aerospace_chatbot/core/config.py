"""Configuration management and environment setup."""

import os, json
import logging
import sys
from pathlib import Path
import warnings
from dotenv import load_dotenv, find_dotenv

class ConfigurationError(Exception):
    """Raised when there are configuration related errors."""
    pass
    
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

def setup_logging():
    """Configure logging based on environment variables.
    
    Environment Variables:
        LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
        LOG_FILE: Path to log file (optional)
        LOG_FORMAT: Custom log format (optional)
    """
    # Suppress Streamlit warning about missing ScriptRunContext
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
    
    # Get settings from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE')
    log_format = os.getenv('LOG_FORMAT', 
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    root_logger.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Always set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optionally set up file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Get the application logger
    logger = logging.getLogger(__name__)
    logger.propagate = False  # Prevent duplicate logs
    
    return logger 