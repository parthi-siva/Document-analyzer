from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
import os
import logging

logger = logging.getLogger(__name__)

class DeepInfraEmbeddings(OpenAIEmbeddings):
    """Custom embeddings class for DeepInfra using LangChain's OpenAI embeddings"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small", 
        **kwargs
    ):
        """
        Initialize DeepInfra embeddings using LangChain's OpenAI embeddings
        
        Args:
            api_key: DeepInfra API key
            model: Embedding model to use
            **kwargs: Additional arguments for OpenAIEmbeddings
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set")
        
        logger.info(f"Initializing DeepInfra embeddings with model: {model}")
        
        super().__init__(
            openai_api_key=api_key,
            openai_api_base="https://api.deepinfra.com/v1/openai",
            model=model,
            **kwargs
        )
        
        logger.debug("DeepInfra embeddings initialized successfully")

# Backward compatibility - keeping the old class name
DeepInfraEmbeddingModel = DeepInfraEmbeddings
