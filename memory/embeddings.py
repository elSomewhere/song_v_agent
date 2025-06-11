"""
Embedding generation for semantic retrieval.
"""
from typing import List, Dict, Optional
import hashlib
from openai import AsyncOpenAI
import asyncio
import os


class EmbeddingGenerator:
    """Generates embeddings for text using OpenAI's API."""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    async def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        tasks = [self.generate(text) for text in texts]
        return await asyncio.gather(*tasks)


class EmbeddingCache:
    """Caches embeddings to avoid redundant API calls."""
    
    def __init__(self, generator: EmbeddingGenerator, max_size: int = 10000):
        self.generator = generator
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.access_counts: Dict[str, int] = {}
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def get_or_generate(self, text: str) -> List[float]:
        """Get embedding from cache or generate if not present."""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.cache:
            self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
            return self.cache[cache_key]
        
        # Generate new embedding
        embedding = await self.generator.generate(text)
        
        # Evict least accessed if at capacity
        if len(self.cache) >= self.max_size:
            least_accessed = min(self.access_counts.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_counts[least_accessed]
        
        # Store new embedding
        self.cache[cache_key] = embedding
        self.access_counts[cache_key] = 1
        
        return embedding
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_counts.clear() 