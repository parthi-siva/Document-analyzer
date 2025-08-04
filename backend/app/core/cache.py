"""
Cache interface and implementations for caching orchestrator results.
Supports both in-memory (development) and Redis (production) backends.
"""

import asyncio
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class CacheInterface(ABC):
    """Abstract interface for caching implementations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL in seconds"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key from function arguments"""
        # Create a string representation of all arguments
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        
        # Create hash for consistent key generation
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"


class InMemoryCache(CacheInterface):
    """Simple in-memory cache implementation for development"""
    
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry['expires_at'] and datetime.now() > entry['expires_at']:
                del self.cache[key]
                return None
            
            return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            # Implement simple LRU eviction if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k]['created_at'])
                del self.cache[oldest_key]
            
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            self.cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': expires_at
            }
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> bool:
        async with self._lock:
            self.cache.clear()
            return True
    
    async def exists(self, key: str) -> bool:
        result = await self.get(key)
        return result is not None


class CacheFactory:
    """Factory for creating cache instances based on configuration"""
    
    @staticmethod
    def create_cache(cache_type: str = "memory", **kwargs) -> CacheInterface:
        """
        Create cache instance based on type
        
        Args:
            cache_type: 'memory' for InMemoryCache, 'redis' for RedisCache
            **kwargs: Additional arguments for cache constructors
        """
        if cache_type.lower() == "memory":
            return InMemoryCache(**kwargs)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")


def cache_result(cache: CacheInterface, key_prefix: str, ttl: Optional[int] = None):
    """
    Decorator to cache function results
    
    Args:
        cache: Cache instance to use
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds (optional)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache.generate_key(key_prefix, *args, **kwargs)
            
            # Try to get from cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for key: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator