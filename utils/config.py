import os
from typing import Dict, Any

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Model configurations
    GEMINI_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Vector store configuration
    COLLECTION_NAME = "documents"
    PERSIST_DIRECTORY = "./chroma_db"
    
    # Retrieval parameters
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Evaluation parameters
    TRULENS_METRICS = ["answer_relevance", "context_relevance", "groundedness", "context_recall"]
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate if required API keys are present"""
        return {
            "gemini": bool(cls.GEMINI_API_KEY),
            "openai": bool(cls.OPENAI_API_KEY)
        }
    
    @classmethod
    def get_chunking_config(cls) -> Dict[str, Any]:
        """Get chunking configuration"""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def get_retrieval_config(cls) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return {
            "top_k": cls.TOP_K,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD
        }
