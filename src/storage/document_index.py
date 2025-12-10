"""In-memory indexing system for O(1) document and chunk lookups - NEW component."""

from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from typing import Dict, Set, Optional, List
import logging

from lib import Config

logger = logging.getLogger(__name__)


class DocumentIndex:
    """
    In-memory indexing system for O(1) document and chunk lookups.
    
    Replaces linear searches with hash-based maps for scalability.
    Only enabled if MCP_INDEXING_ENABLED=true.
    
    Based on reference implementation but adapted for Python and your architecture.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize DocumentIndex.

        Args:
            data_dir: Data directory where index file will be stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory maps for O(1) lookups
        self.document_map: Dict[str, str] = {}  # document_id -> filepath
        self.chunk_map: Dict[str, Dict[str, str | int]] = {}  # chunk_id -> {doc_id, chunk_index}
        self.content_hash: Dict[str, str] = {}  # content_hash -> document_id (deduplication)
        self.keyword_index: Dict[str, Set[str]] = {}  # keyword -> set of document_ids
        
        self.index_file = self.data_dir / "document-index.json"
        self.initialized = False
        
        logger.info(f"DocumentIndex initialized (index file: {self.index_file})")

    def initialize(self) -> None:
        """
        Initialize the index by loading from disk or building from existing documents.
        
        Should be called after DocumentStore is ready.
        """
        if self.initialized:
            return

        if not Config.INDEXING_ENABLED:
            logger.info("DocumentIndex disabled (MCP_INDEXING_ENABLED=false)")
            return

        try:
            # Try to load existing index
            self.load_index()
            logger.info(f"DocumentIndex loaded from disk: {len(self.document_map)} documents")
        except Exception as e:
            logger.warning(f"Failed to load existing index, will rebuild on demand: {e}")
            # Don't rebuild here - let it happen lazily when needed

        self.initialized = True

    def add_document(
        self,
        document_id: str,
        filepath: str,
        content: str,
        chunks: Optional[List] = None,
    ) -> None:
        """
        Add a document to the index.
        
        Args:
            document_id: Document ID
            filepath: Path to document file (or virtual path)
            content: Document content (for hashing and keyword extraction)
            chunks: Optional list of chunks (for chunk indexing)
        """
        if not Config.INDEXING_ENABLED:
            return

        # Add to document map
        self.document_map[document_id] = filepath

        # Add content hash for deduplication
        content_hash_value = self.hash_content(content)
        self.content_hash[content_hash_value] = document_id

        # Add chunks to chunk map
        if chunks:
            for idx, chunk in enumerate(chunks):
                chunk_id = getattr(chunk, 'chunk_id', None) or getattr(chunk, 'id', None)
                if chunk_id:
                    self.chunk_map[chunk_id] = {
                        "doc_id": document_id,
                        "chunk_index": idx
                    }

        # Extract and index keywords
        self.index_keywords(document_id, content)

        # Persist index (async, don't block)
        try:
            self.save_index()
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")

    def remove_document(self, document_id: str) -> None:
        """
        Remove a document from the index.
        
        Args:
            document_id: Document ID to remove
        """
        if not Config.INDEXING_ENABLED:
            return

        filepath = self.document_map.get(document_id)
        if not filepath:
            return

        # Remove from document map
        self.document_map.pop(document_id, None)

        # Remove from content hash
        hashes_to_remove = [
            hash_val for hash_val, doc_id in self.content_hash.items()
            if doc_id == document_id
        ]
        for hash_val in hashes_to_remove:
            self.content_hash.pop(hash_val, None)

        # Remove chunks
        chunks_to_remove = [
            chunk_id for chunk_id, chunk_info in self.chunk_map.items()
            if chunk_info.get("doc_id") == document_id
        ]
        for chunk_id in chunks_to_remove:
            self.chunk_map.pop(chunk_id, None)

        # Remove from keyword index
        for keyword, doc_ids in self.keyword_index.items():
            doc_ids.discard(document_id)
            if len(doc_ids) == 0:
                self.keyword_index.pop(keyword, None)

        # Persist index
        try:
            self.save_index()
        except Exception as e:
            logger.warning(f"Failed to save index after removal: {e}")

    def find_document(self, document_id: str) -> Optional[str]:
        """
        Find document file path by ID - O(1) lookup.
        
        Args:
            document_id: Document ID
            
        Returns:
            File path if found, None otherwise
        """
        if not Config.INDEXING_ENABLED:
            return None
        return self.document_map.get(document_id)

    def find_chunk(self, chunk_id: str) -> Optional[Dict[str, str | int]]:
        """
        Find chunk information by chunk ID - O(1) lookup.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Dict with doc_id and chunk_index if found, None otherwise
        """
        if not Config.INDEXING_ENABLED:
            return None
        return self.chunk_map.get(chunk_id)

    def find_duplicate_content(self, content: str) -> Optional[str]:
        """
        Find duplicate content by hash - O(1) lookup.
        
        Args:
            content: Content to check for duplicates
            
        Returns:
            Document ID of duplicate if found, None otherwise
        """
        if not Config.INDEXING_ENABLED:
            return None
        content_hash_value = self.hash_content(content)
        return self.content_hash.get(content_hash_value)

    def search_by_keywords(self, keywords: List[str]) -> Set[str]:
        """
        Search documents by keywords - much faster than full text search.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Set of document IDs matching all keywords
        """
        if not Config.INDEXING_ENABLED or not keywords:
            return set()

        # Start with first keyword
        result = self.keyword_index.get(keywords[0].lower(), set()).copy()
        
        # Intersect with remaining keywords
        for keyword in keywords[1:]:
            keyword_docs = self.keyword_index.get(keyword.lower(), set())
            result = result.intersection(keyword_docs)
            if not result:
                break  # No documents match all keywords
        
        return result

    def get_all_document_ids(self) -> List[str]:
        """
        Get all document IDs - O(1) size, O(n) iteration.
        
        Returns:
            List of all document IDs
        """
        if not Config.INDEXING_ENABLED:
            return []
        return list(self.document_map.keys())

    def get_stats(self) -> Dict[str, int]:
        """
        Get index statistics.
        
        Returns:
            Dict with counts of documents, chunks, and keywords
        """
        return {
            "documents": len(self.document_map),
            "chunks": len(self.chunk_map),
            "keywords": len(self.keyword_index),
            "enabled": Config.INDEXING_ENABLED,
        }

    def hash_content(self, content: str) -> str:
        """
        Hash content for deduplication.
        
        Args:
            content: Content to hash
            
        Returns:
            Hash string (first 16 chars of SHA256)
        """
        return hashlib.sha256(content.strip().encode()).hexdigest()[:16]

    def index_keywords(self, document_id: str, content: str) -> None:
        """
        Extract and index keywords from content.
        
        Args:
            document_id: Document ID
            content: Document content
        """
        keywords = self.extract_keywords(content)
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.keyword_index:
                self.keyword_index[keyword_lower] = set()
            self.keyword_index[keyword_lower].add(document_id)

    def extract_keywords(self, content: str) -> List[str]:
        """
        Extract keywords from text (simple word extraction).
        
        Args:
            content: Text content
            
        Returns:
            List of unique keywords
        """
        import re
        words = re.sub(r'[^\w\s]', ' ', content.lower()).split()
        
        # Filter: length 3-20, not stop words
        keywords = [
            word for word in words
            if 3 <= len(word) <= 20 and not self.is_stop_word(word)
        ]
        
        # Return unique words
        return list(set(keywords))

    def is_stop_word(self, word: str) -> bool:
        """
        Check if word is a stop word (basic list).
        
        Args:
            word: Word to check
            
        Returns:
            True if stop word, False otherwise
        """
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        return word in stop_words

    def save_index(self) -> None:
        """Save index to disk."""
        if not Config.INDEXING_ENABLED:
            return

        index_data = {
            "version": "1.0",
            "document_map": self.document_map,
            "chunk_map": self.chunk_map,
            "content_hash": self.content_hash,
            "keyword_index": {
                key: list(value) for key, value in self.keyword_index.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        try:
            with open(self.index_file, "w") as f:
                json.dump(index_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self) -> None:
        """Load index from disk."""
        if not self.index_file.exists():
            raise FileNotFoundError("Index file does not exist")

        try:
            with open(self.index_file) as f:
                index_data = json.load(f)
            
            self.document_map = index_data.get("document_map", {})
            self.chunk_map = index_data.get("chunk_map", {})
            self.content_hash = index_data.get("content_hash", {})
            
            # Convert keyword_index lists back to sets
            keyword_data = index_data.get("keyword_index", {})
            self.keyword_index = {
                key: set(value) for key, value in keyword_data.items()
            }
            
            logger.info(f"Loaded index: {len(self.document_map)} documents")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise