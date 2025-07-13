import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for production settings."""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    
    # Application Settings
    APP_ENV = os.getenv('APP_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Model Configuration
    MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-1.5-flash')
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
    MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '1024'))
    TOP_P = float(os.getenv('TOP_P', '0.8'))
    TOP_K = int(os.getenv('TOP_K', '40'))
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    # Retrieval Configuration
    RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '5'))
    SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', '0.7'))
    
    # Security Settings
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', '500'))
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))
    
    # Cache Settings
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))
    MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '100'))
    
    # Pinecone Settings
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'pakistan-law')
    PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '384'))  # for all-MiniLM-L6-v2
    PINECONE_METRIC = os.getenv('PINECONE_METRIC', 'cosine')
    PINECONE_BATCH_SIZE = int(os.getenv('PINECONE_BATCH_SIZE', '100'))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required")
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required")
        return True
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            'app_env': cls.APP_ENV,
            'debug': cls.DEBUG,
            'log_level': cls.LOG_LEVEL,
            'model': {
                'name': cls.MODEL_NAME,
                'temperature': cls.TEMPERATURE,
                'max_output_tokens': cls.MAX_OUTPUT_TOKENS,
                'top_p': cls.TOP_P,
                'top_k': cls.TOP_K
            },
            'embedding': {
                'model': cls.EMBEDDING_MODEL,
                'chunk_size': cls.CHUNK_SIZE,
                'chunk_overlap': cls.CHUNK_OVERLAP
            },
            'retrieval': {
                'k': cls.RETRIEVAL_K,
                'score_threshold': cls.SCORE_THRESHOLD
            },
            'security': {
                'max_input_length': cls.MAX_INPUT_LENGTH,
                'rate_limit_requests': cls.RATE_LIMIT_REQUESTS,
                'rate_limit_window': cls.RATE_LIMIT_WINDOW
            }
        }