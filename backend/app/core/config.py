"""
Configuration settings for the document analyzer application.
Includes cache configuration and other application settings.
"""

import os


class CacheConfig:
    """Cache configuration settings"""

    # Cache type: 'memory' for development, 'redis' for production
    CACHE_TYPE: str = os.getenv("CACHE_TYPE", "memory")

    # Cache TTL settings (in seconds)
    DEFAULT_CACHE_TTL: int = int(os.getenv("DEFAULT_CACHE_TTL", "3600"))  # 1 hour
    DOCUMENT_PROCESS_CACHE_TTL: int = int(
        os.getenv("DOCUMENT_PROCESS_CACHE_TTL", "7200")
    )  # 2 hours
    QUERY_CACHE_TTL: int = int(os.getenv("QUERY_CACHE_TTL", "1800"))  # 30 minutes

    # In-memory cache settings
    MEMORY_CACHE_MAX_SIZE: int = int(os.getenv("MEMORY_CACHE_MAX_SIZE", "1000"))

    # Cache behavior settings
    ENABLE_DOCUMENT_PROCESSING_CACHE: bool = (
        os.getenv("ENABLE_DOCUMENT_PROCESSING_CACHE", "true").lower() == "true"
    )
    ENABLE_QUERY_CACHE: bool = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"


class AppConfig:
    """General application configuration"""

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Cache configuration
    cache = CacheConfig()


# Global config instance
config = AppConfig()
