from typing import List, Optional, Callable, Any
import logging
from collections import OrderedDict
import hashlib
from lancedb.embeddings import get_registry
from lib import Config

logger = logging.getLogger(__name__)

"""LRU cache for embeddings"""
class EmbeddingCache:
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    # Get embedding from cache
    def get(self, text: str) -> Optional[List[float]]:
        key = self.hash_text(text)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    # Store embedding in cache
    def set(self,text: str, embedding: List[float]):
        key = self.hash_text(text)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            self.cache[key] = embedding
            self.cache.move_to_end(key)

    def hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

"""Handles embedding with caching"""
class Embedder:
    def __init__(self, default_model: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.default_model = default_model
        self.embedding_functions: dict[str, Any] = {}
        self.cache: Optional[EmbeddingCache] = None

        if Config.CACHE_ENABLED:
            self.cache = EmbeddingCache(max_size=Config.CACHE_SIZE)
            logger.info(f"Embedding cache enabled (size: {Config.CACHE_SIZE})")
        
        logger.info(f"Embedder intiailized with default model: {default_model}")

    def get_embedding_function(self, model_name: Optional[str] = None) -> Any:
        """Get or create embedding function for a model."""
        model = model_name or self.default_model

        if model in self.embedding_functions:
            return self.embedding_functions[model]

        try:
            embedding_func = get_registry().get("huggingface").create(name=model)
            
            # Create a callable wrapper function
            # TransformersEmbeddingFunction objects may not be directly callable
            def wrapper(texts):
                """Wrapper to make embedding function callable."""
                try:
                    # Try calling directly first
                    return embedding_func(texts)  # type: ignore[operator]
                except TypeError as e:
                    error_msg = str(e)
                    if "'TransformersEmbeddingFunction' object is not callable" in error_msg or "not callable" in error_msg.lower():
                        # Try alternative methods
                        # Method 1: Try embed method
                        if hasattr(embedding_func, 'embed'):
                            try:
                                return embedding_func.embed(texts)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        
                        # Method 2: Try accessing underlying model with encode
                        model_attr = None
                        for attr_name in ['_model', 'model', 'encoder', '_encoder']:
                            if hasattr(embedding_func, attr_name):
                                model_attr = getattr(embedding_func, attr_name)
                                break
                        
                        if model_attr:
                            # Try encode method (common in sentence-transformers)
                            if hasattr(model_attr, 'encode'):
                                try:
                                    result = model_attr.encode(texts if isinstance(texts, list) else [texts])
                                    # Convert to list if needed
                                    if hasattr(result, 'tolist'):
                                        result = result.tolist()  # type: ignore[attr-defined]
                                    elif hasattr(result, '__iter__') and not isinstance(result, (list, tuple)):
                                        result = list(result)
                                    # If single text was passed, return single embedding
                                    if not isinstance(texts, list) and isinstance(result, list) and len(result) == 1:
                                        return result[0]
                                    return result
                                except Exception as encode_error:
                                    logger.debug(f"encode method failed: {encode_error}")
                            
                            # Try calling model directly
                            if callable(model_attr):
                                try:
                                    return model_attr(texts)
                                except Exception:
                                    pass
                        
                        # Method 3: Try compute_source_embeddings (LanceDB embedding function API)
                        if hasattr(embedding_func, 'compute_source_embeddings'):
                            try:
                                # compute_source_embeddings expects a list of texts
                                result = embedding_func.compute_source_embeddings(  # type: ignore[attr-defined]
                                    texts if isinstance(texts, list) else [texts]
                                )
                                # Convert to list format if needed
                                if hasattr(result, 'tolist'):
                                    result = result.tolist()  # type: ignore[attr-defined]
                                elif hasattr(result, '__iter__') and not isinstance(result, (list, tuple)):
                                    result = list(result)
                                # If single text was passed, return single embedding
                                if not isinstance(texts, list) and isinstance(result, list) and len(result) == 1:
                                    return result[0]
                                return result
                            except Exception as compute_error:
                                logger.debug(f"compute_source_embeddings failed: {compute_error}")
                        
                        # Method 4: Try source attribute
                        if hasattr(embedding_func, 'source'):
                            source_func = embedding_func.source  # type: ignore[attr-defined]
                            if callable(source_func):
                                try:
                                    return source_func(texts)
                                except Exception:
                                    pass
                        
                        # If all methods failed, raise informative error
                        available_attrs = [attr for attr in dir(embedding_func) if not attr.startswith('_')]
                        raise ValueError(
                            f"Embedding function for {model} is not callable. "
                            f"Tried: direct call, embed(), model.encode(), model(), compute_source_embeddings(), source(). "
                            f"Available public attributes: {available_attrs[:10]}. "
                            f"Consider updating lancedb or checking the embedding function API."
                        ) from e
                    # Re-raise if it's a different TypeError
                    raise
            
            func = wrapper
            self.embedding_functions[model] = func
            logger.info(f"Created embedding function for model: {model}")
            return func
        except Exception as e:
            logger.error(f"Failed to create embedding function for {model}: {e}")
            raise

    def embed_text(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """Generate embedding for a single text with caching."""
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text")
            dim = self.get_embedding_dimension(model_name)
            return [0.0] * dim

        # Check cache first
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        try:
            func = self.get_embedding_function(model_name)
            embeddings = func([text])
            embedding = embeddings[0]
            
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding)

            # Store in cache
            if self.cache:
                self.cache.set(text, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with caching."""
        if not texts:
            return []

        # Filter out empty texts
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            dim = self.get_embedding_dimension(model_name)
            return [[0.0] * dim] * len(texts)

        # Check cache for all texts
        cached_embeddings = {}
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(valid_texts):
            if self.cache:
                cached = self.cache.get(text)
                if cached is not None:
                    cached_embeddings[i] = cached
                    continue
            texts_to_embed.append(text)
            text_indices.append(i)

        # Embed texts not in cache
        embeddings = []
        if texts_to_embed:
            try:
                func = self.get_embedding_function(model_name)
                total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size

                for batch_idx in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[batch_idx:batch_idx + batch_size]
                    
                    if progress_callback:
                        progress_callback(
                            batch_idx // batch_size + 1,
                            total_batches,
                            f"Embedding batch {batch_idx // batch_size + 1}/{total_batches}"
                        )

                    batch_embeddings = func(batch)
                    
                    for emb in batch_embeddings:
                        if hasattr(emb, "tolist"):
                            embeddings.append(emb.tolist())
                        elif isinstance(emb, list):
                            embeddings.append(emb)
                        else:
                            embeddings.append(list(emb))

                # Store in cache
                if self.cache:
                    for text, embedding in zip(texts_to_embed, embeddings):
                        self.cache.set(text, embedding)

            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                raise

        # Reconstruct full result
        result: List[List[float]] = [[] for _ in range(len(valid_texts))]
        for i, idx in enumerate(valid_indices):
            if idx in cached_embeddings:
                result[i] = cached_embeddings[idx]
            else:
                result[i] = embeddings[text_indices.index(idx)]

        # Fill in empty embeddings for filtered texts
        if len(result) < len(texts):
            dim = len(result[0]) if result and len(result[0]) > 0 else self.get_embedding_dimension(model_name)
            empty_emb = [0.0] * dim
            full_result: List[List[float]] = [empty_emb for _ in range(len(texts))]
            for idx, emb in zip(valid_indices, result):
                if len(emb) > 0:
                    full_result[idx] = emb
            return full_result

        return result

    def get_embedding_dimension(self, model_name: Optional[str] = None) -> int:
        """Get the dimension of embeddings for a model."""
        try:
            func = self.get_embedding_function(model_name)
            if hasattr(func, 'ndims'):
                return func.ndims()
            dummy_embeddings = func(["dummy"])
            if dummy_embeddings and len(dummy_embeddings) > 0:
                embedding = dummy_embeddings[0]
                if hasattr(embedding, '__len__'):
                    return len(embedding)
            return 768
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            return 768

    def get_cache_stats(self) -> Optional[dict]:
        """Get cache statistics."""
        return self.cache.get_stats() if self.cache else None