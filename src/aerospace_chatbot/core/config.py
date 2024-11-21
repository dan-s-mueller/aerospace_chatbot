"""Configuration management and environment setup."""

import os, json
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
    """Configure logging based on environment variables."""
    import warnings
    import os
    import sys
    from pathlib import Path
    import logging

    # Suppress Streamlit warning about missing ScriptRunContext
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
    
    # Get settings from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE')
    log_format = os.getenv('LOG_FORMAT', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Reset all existing loggers
    logging.shutdown()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Disable existing loggers
    logging.getLogger('aerospace_chatbot').handlers.clear()
    
    # Configure the root logger first
    root_logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Configure the aerospace_chatbot logger
    logger = logging.getLogger('aerospace_chatbot')
    logger.setLevel(numeric_level)
    logger.propagate = True  # Allow propagation to root logger
    
    # Explicitly configure test logger if we're in a test environment
    if 'pytest' in sys.modules:
        test_logger = logging.getLogger('tests')
        test_logger.setLevel(numeric_level)
        test_logger.propagate = True
    
    # Print confirmation of logging setup
    root_logger.info("Logging configured successfully")
    
    return logger 

def get_required_api_keys(config, selected_services):
    """Determine required API keys based on config and selected services."""
    required_keys = {}
    
    # Map services to their required environment variables
    service_key_mapping = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Hugging Face': 'HUGGINGFACEHUB_API_KEY',
        'Voyage': 'VOYAGE_API_KEY',
        'Pinecone': 'PINECONE_API_KEY'
    }
    
    # Check database requirements
    for db in config['databases']:
        if db['name'] == selected_services.get('index_type'):
            if db['name'] in service_key_mapping:
                required_keys[db['name']] = service_key_mapping[db['name']]
    
    # Check embedding service requirements
    if selected_services.get('embedding_service') in service_key_mapping:
        required_keys[selected_services['embedding_service']] = service_key_mapping[selected_services['embedding_service']]
    
    # Check LLM service requirements
    if selected_services.get('llm_service') in service_key_mapping:
        required_keys[selected_services['llm_service']] = service_key_mapping[selected_services['llm_service']]
    
    return required_keys 