from typing import Callable, Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
import logging
import asyncio
from datetime import datetime, timezone

from lib import (
    Config,
    CollectionNotFoundError,
    DocumentNotFoundError,
    CollectionNotEmptyError,
    CollectionAlreadyExistsError,
)
from storage import CollectionManager, DocumentStore, VectorStore
from .indexing import IndexingPipeline

if TYPE_CHECKING:
    from retrieval import SearchEngine

logger = logging.getLogger(__name__)


class DocumentManager:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        # Dependency injection (for testing)
        collection_manager: Optional[CollectionManager] = None,
        document_store: Optional[DocumentStore] = None,
        vector_store: Optional[VectorStore] = None,
        indexing_pipeline: Optional[IndexingPipeline] = None,
        search_engine: Optional['SearchEngine'] = None,
    ):
        self.base_dir = base_dir or Config.BASE_DIR
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL

        self.data_dir = self.base_dir / "data"
        self.collections_dir = self.data_dir / "collections"
        self.uploads_dir = self.base_dir/"uploads"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collections_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # Store injected components
        self.collection_manager = collection_manager
        self.document_store = document_store
        self.vector_store = vector_store
        self.indexing_pipeline = indexing_pipeline
        self.search_engine = search_engine

        self.initialized = False

        logger.info("DocumentManager created")

    def ensure_initialized(self):
        if self.initialized:
            return
        
        logger.info("Initializing DocumentManager components...")

        # Initialize components with dependency injection support
        if self.collection_manager is None:
            self.collection_manager = CollectionManager(
                data_dir=str(self.collections_dir)
            )

        if self.document_store is None:
            from storage import DocumentStore
            self.document_store = DocumentStore(
                db_path=str(self.data_dir / "documents.db")
            )
        
        if self.vector_store is None:
            from storage import VectorStore
            from core import Embedder
            embedder = Embedder(default_model=self.embedding_model)
            self.vector_store = VectorStore(
                db_path=str(self.data_dir / "lancedb"),
                embedder=embedder,
                default_embedding_model=self.embedding_model
            )
        
        if self.indexing_pipeline is None:
            self.indexing_pipeline = IndexingPipeline(
                collection_manager=self.collection_manager,
                document_store=self.document_store,
                vector_store=self.vector_store,
            )
        
        if self.search_engine is None:
            from retrieval import SearchEngine
            # Pass document_index if available for keyword search support
            document_index = None
            if self.indexing_pipeline and hasattr(self.indexing_pipeline, 'document_index'):
                document_index = self.indexing_pipeline.document_index
            self.search_engine = SearchEngine(
                vector_store=self.vector_store,
                document_store=self.document_store,
                document_index=document_index,
            )
        
        self.initialized = True
        logger.info("DocumentManager initialized successfully")

    """Collection Management"""

    def create_collection(
        self,
        collection_id: str,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        
        self.ensure_initialized()
        assert self.collection_manager is not None
        
        collection = self.collection_manager.create_collection(
            collection_id=collection_id,
            description=description,
            embedding_model=embedding_model or self.embedding_model,
            chunk_size=chunk_size or Config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP,
        )
        return {
            "collection_id": collection.collection_id,
            "status": "created",
            "embedding_model": collection.embedding_model,
            "chunk_size": collection.chunk_size,
            "chunk_overlap": collection.chunk_overlap,
        }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        assert self.collection_manager

        collections = self.collection_manager.list_collections()
        return [
            {
                "id": c.collection_id,
                "description": c.description,
                "document_count": c.document_count,
                "chunk_count": c.chunk_count,
                "last_updated": c.last_updated.isoformat(),
            }
            for c in collections
        ]

    def delete_collection(
        self,
        collection_id: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete a collection and optionally all its documents.
        
        Args:
            collection_id: Collection ID to delete
            force: If True, delete even if collection has documents
            
        Returns:
            Dictionary with deletion result
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            CollectionNotEmptyError: If collection has documents and force=False
        """
        self.ensure_initialized()
        assert self.collection_manager
        assert self.indexing_pipeline
        assert self.document_store
        assert self.vector_store
        
        # Check if collection exists
        collection = self.collection_manager.get_collection(collection_id)
        if not collection:
            available = [c.collection_id for c in self.collection_manager.list_collections()]
            raise CollectionNotFoundError(collection_id, available)
        
        # Check if collection has documents
        docs = self.document_store.get_documents_by_collection(collection_id)
        if docs and not force:
            raise CollectionNotEmptyError(collection_id, len(docs))
        
        # Delete all documents in the collection
        deleted_docs = 0
        deleted_chunks = 0
        for doc in docs:
            # Delete chunks from vector store
            chunks_deleted = self.vector_store.delete_chunks_by_document(
                document_id=doc.document_id,
                collection_id=collection_id,
            )
            deleted_chunks += chunks_deleted
            
            # Delete document metadata
            self.document_store.delete_document(doc.document_id)
            deleted_docs += 1
        
        # Delete the collection itself
        self.collection_manager.delete_collection(collection_id)
        
        # Try to drop the vector table
        try:
            table_name = self.vector_store.get_table_name(collection_id)
            if table_name in self.vector_store.db.table_names():
                self.vector_store.db.drop_table(table_name)
        except Exception as e:
            logger.warning(f"Failed to drop vector table for collection {collection_id}: {e}")
        
        logger.info(f"Deleted collection {collection_id} with {deleted_docs} documents and {deleted_chunks} chunks")
        
        return {
            "collection_id": collection_id,
            "status": "deleted",
            "documents_deleted": deleted_docs,
            "chunks_deleted": deleted_chunks,
        }

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a collection.
        
        Args:
            collection_id: Collection ID
            
        Returns:
            Collection details or None if not found
        """
        self.ensure_initialized()
        assert self.collection_manager
        
        collection = self.collection_manager.get_collection(collection_id)
        if not collection:
            return None
        
        return {
            "id": collection.collection_id,
            "description": collection.description,
            "embedding_model": collection.embedding_model,
            "chunk_size": collection.chunk_size,
            "chunk_overlap": collection.chunk_overlap,
            "document_count": collection.document_count,
            "chunk_count": collection.chunk_count,
            "created_at": collection.created_at.isoformat(),
            "last_updated": collection.last_updated.isoformat(),
        }
    
    """Document Management"""

    def index_document_from_source(
        self,
        source: str,
        collection_id: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        
        self.ensure_initialized()
        assert self.indexing_pipeline

        result = self.indexing_pipeline.index_document(
            source=source,
            collection_id=collection_id,
            document_type=document_type,
            custom_metadata=metadata,
            force_reindex=force_reindex,
            progress_callback=progress_callback,
        )

        if result.status == "indexed":
            assert self.document_store
            doc = self.document_store.get_document(result.document_id)
            if doc is not None:
                return {
                    "document_id": result.document_id,
                    "collection_id": result.collection_id,
                    "source": doc.source_url or source,
                    "status": "indexed",
                    "chunk_count": result.chunk_count,
                    "pages": doc.pages,
                    "indexed_at": doc.indexed_at.isoformat(),
                    "metadata": {
                        "filename": doc.filename,
                        "title": doc.title,
                        "document_type": doc.document_type,
                    }
                }
        return {
            "document_id": result.document_id,
            "status": result.status,
            "error": result.error
        }
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        self.ensure_initialized()

        assert self.document_store
        assert self.vector_store

        doc = self.document_store.get_document(document_id)    
        if not doc:
            return None
        
        chunk_count = self.vector_store.count_chunks(
            collection_id=doc.collection_id,
            document_id=document_id
        )

        return {
            "id": doc.document_id,
            "title": doc.title,
            "collection_id": doc.collection_id,
            "metadata": doc.custom_metadata,
            "created_at": doc.indexed_at.isoformat(),
            "chunks_count": chunk_count,
        }
    
    def get_all_documents(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure_initialized()

        assert self.document_store
        
        if collection_id:
            docs = self.document_store.get_documents_by_collection(collection_id)
        else:
            docs = self.document_store.list_documents()
        
        return [
            {
                "id": doc.document_id,
                "title": doc.title,
                "collection_id": doc.collection_id,
                "created_at": doc.indexed_at.isoformat(),
                "metadata": doc.custom_metadata,
            }
            for doc in docs
        ]
    
    def delete_document(self, document_id: str) -> bool:
        self.ensure_initialized()
        assert self.indexing_pipeline
        return self.indexing_pipeline.delete_document(document_id)

    def update_document_metadata(
        self,
        document_id: str,
        title: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update document metadata without re-indexing.
        
        Args:
            document_id: Document ID to update
            title: New title (optional)
            custom_metadata: New custom metadata to merge (optional)
            
        Returns:
            Updated document details
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        self.ensure_initialized()
        assert self.document_store
        
        # Get existing document
        doc = self.document_store.get_document(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)
        
        # Update fields if provided
        if title is not None:
            doc.title = title
        
        if custom_metadata is not None:
            # Merge with existing metadata
            doc.custom_metadata.update(custom_metadata)
        
        # Save updated document
        self.document_store.update_document(doc)
        
        logger.info(f"Updated metadata for document {document_id}")
        
        return {
            "document_id": document_id,
            "status": "updated",
            "title": doc.title,
            "custom_metadata": doc.custom_metadata,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    """Search"""

    def search_documents(
        self,
        query: str,
        collection_ids: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        top_k: int = 10,
        include_context: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search documents.
        
        Args:
            query: Search query
            collection_ids: Optional list of collection IDs to search
            document_id: Optional document ID to search within
            top_k: Number of results to return
            include_context: If True, include adjacent chunks as context
            
        Returns:
            List of search result dictionaries
        """
        self.ensure_initialized()
        assert self.search_engine
        
        if document_id:
            results = self.search_engine.search_by_document(
                document_id=document_id,
                query=query,
                top_k=top_k,
            )
        else:
            results = self.search_engine.search(
                query=query,
                collection_ids=collection_ids,
                top_k=top_k,
                include_context=include_context,
            )
        
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "document_id": r.chunk.metadata.document_id,
                "collection_id": r.chunk.metadata.collection_id,
                "chunk_index": r.chunk.metadata.chunk_index,
                "score": r.score,
                "text": r.chunk.text,
                "metadata": {
                    "filename": r.chunk.metadata.filename,
                    "page_numbers": r.chunk.metadata.page_numbers,
                    "section": r.chunk.metadata.section_hierarchy[-1] if r.chunk.metadata.section_hierarchy else None,
                    "headings": r.chunk.metadata.section_hierarchy,
                },
                "context": {
                    "before": r.context_before,
                    "after": r.context_after,
                } if include_context else None,
            }
            for r in results
        ]
    
    def get_context_window(
        self,
        document_id: str,
        chunk_index: int,
        before: int = 1,
        after: int = 1,
    ) -> Dict[str, Any]:
        """
        Get context window around a chunk.
        
        Args:
            document_id: Document ID
            chunk_index: Chunk index to center on
            before: Number of chunks before
            after: Number of chunks after
            
        Returns:
            Dictionary with context window information
        """
        self.ensure_initialized()
        assert self.document_store
        assert self.search_engine
        
        doc = self.document_store.get_document(document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found")
        
        # Get all chunks for this document using the existing search engine
        chunks_with_scores = self.search_engine.search_with_scores(
            query="",  # Empty query to get all chunks
            collection_id=doc.collection_id,
            top_k=10000,  # Large number to get all chunks
            filters={"document_id": document_id},
        )
        
        # Extract chunks and sort by index
        chunks = [c for c, _ in chunks_with_scores]
        chunks.sort(key=lambda c: c.metadata.chunk_index)
        
        # Find the chunk at the specified index
        target_chunk = None
        target_pos = None
        for i, chunk in enumerate(chunks):
            if chunk.metadata.chunk_index == chunk_index:
                target_chunk = chunk
                target_pos = i
                break
        
        if target_chunk is None or target_pos is None:
            raise ValueError(f"Chunk at index {chunk_index} not found in document {document_id}")
        
        # Get context window
        total = len(chunks)
        start = max(0, target_pos - before)
        end = min(total, target_pos + after + 1)
        
        window_chunks = [
            {
                "chunk_index": c.metadata.chunk_index,
                "text": c.text,
            }
            for c in chunks[start:end]
        ]
        
        return {
            "window": window_chunks,
            "center": chunk_index,
            "total_chunks": total,
        }
    
    # File Upload Support
    def get_uploads_dir(self) -> str:
        return str(self.uploads_dir.resolve())

    def process_uploads_folder(self) -> Dict[str, Any]:
        self.ensure_initialized()
        # Implementation would scan and process files
        return {"processed": 0, "errors": []}

    def list_uploads_files(self) -> List[Dict[str, Any]]:
        files = []
        for file_path in self.uploads_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "supported": file_path.suffix.lower() in [".txt", ".md", ".pdf"],
                })
        return files

    def get_data_dir(self) -> str:
        return str(self.data_dir.resolve())

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system-wide statistics
        """
        self.ensure_initialized()
        assert self.collection_manager
        assert self.document_store
        assert self.vector_store
        assert self.indexing_pipeline
        
        # Get collection stats
        collections = self.collection_manager.list_collections()
        total_collections = len(collections)
        total_documents = sum(c.document_count for c in collections)
        total_chunks = sum(c.chunk_count for c in collections)
        
        # Get storage stats
        storage_stats = self._get_storage_stats()
        
        # Get embedding cache stats if available
        cache_stats = {}
        if hasattr(self.vector_store, 'embedder') and hasattr(self.vector_store.embedder, 'get_cache_stats'):
            try:
                cache_stats = self.vector_store.embedder.get_cache_stats()
            except Exception:
                pass
        
        # Get document index stats if available
        index_stats = {}
        if hasattr(self.indexing_pipeline, 'document_index') and self.indexing_pipeline.document_index:
            try:
                index_stats = self.indexing_pipeline.document_index.get_stats()
            except Exception:
                pass
        
        return {
            "collections": {
                "total": total_collections,
                "list": [
                    {
                        "id": c.collection_id,
                        "documents": c.document_count,
                        "chunks": c.chunk_count,
                    }
                    for c in collections
                ],
            },
            "documents": {
                "total": total_documents,
            },
            "chunks": {
                "total": total_chunks,
            },
            "storage": storage_stats,
            "embedding_cache": cache_stats,
            "document_index": index_stats,
            "config": {
                "embedding_model": self.embedding_model,
                "default_chunk_size": Config.DEFAULT_CHUNK_SIZE,
                "default_chunk_overlap": Config.DEFAULT_CHUNK_OVERLAP,
                "cache_enabled": Config.CACHE_ENABLED,
                "parallel_enabled": Config.PARALLEL_ENABLED,
            },
        }

    def _get_storage_stats(self) -> Dict[str, Any]:
        """Get storage-related statistics."""
        import os
        
        def get_dir_size(path: Path) -> int:
            """Get total size of directory in bytes."""
            total = 0
            try:
                for entry in path.rglob("*"):
                    if entry.is_file():
                        total += entry.stat().st_size
            except Exception:
                pass
            return total
        
        def format_size(size_bytes: float) -> str:
            """Format size in human-readable format."""
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} TB"
        
        # Get sizes
        data_size = get_dir_size(self.data_dir)
        lancedb_path = self.data_dir / "lancedb"
        lancedb_size = get_dir_size(lancedb_path) if lancedb_path.exists() else 0
        
        db_path = self.data_dir / "documents.db"
        db_size = db_path.stat().st_size if db_path.exists() else 0
        
        return {
            "data_directory": str(self.data_dir),
            "total_size_bytes": data_size,
            "total_size_formatted": format_size(data_size),
            "vector_store_bytes": lancedb_size,
            "vector_store_formatted": format_size(lancedb_size),
            "document_db_bytes": db_size,
            "document_db_formatted": format_size(db_size),
        }

    def hybrid_search(
        self,
        query: str,
        collection_ids: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        top_k: int = 10,
        include_context: bool = False,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        rerank_method: str = "combined",
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Args:
            query: Search query
            collection_ids: Optional list of collection IDs to search
            document_id: Optional document ID to search within
            top_k: Number of results to return
            include_context: If True, include adjacent chunks as context
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            keyword_weight: Weight for keyword matching (0.0 to 1.0)
            rerank: If True, apply reranking to results
            rerank_method: Reranking method ("keyword_boost", "length_penalty", "position_boost", "combined")
            
        Returns:
            List of search result dictionaries
        """
        self.ensure_initialized()
        assert self.search_engine
        
        # Use hybrid search
        results = self.search_engine.hybrid_search(
            query=query,
            collection_ids=collection_ids,
            top_k=top_k if not rerank else top_k * 2,  # Get more for reranking
            include_context=include_context,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )
        
        # Apply reranking if requested
        if rerank and results:
            results = self.search_engine.rerank_results(
                query=query,
                results=results,
                top_k=top_k,
                method=rerank_method,
            )
        
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "document_id": r.chunk.metadata.document_id,
                "collection_id": r.chunk.metadata.collection_id,
                "chunk_index": r.chunk.metadata.chunk_index,
                "score": r.score,
                "text": r.chunk.text,
                "metadata": {
                    "filename": r.chunk.metadata.filename,
                    "page_numbers": r.chunk.metadata.page_numbers,
                    "section": r.chunk.metadata.section_hierarchy[-1] if r.chunk.metadata.section_hierarchy else None,
                    "headings": r.chunk.metadata.section_hierarchy,
                },
                "context": {
                    "before": r.context_before,
                    "after": r.context_after,
                } if include_context else None,
            }
            for r in results
        ]


class ProgressCallback:
    """
    Async-compatible progress callback for long-running operations.
    
    Can be used with MCP context for real-time progress reporting.
    """
    
    def __init__(self, ctx=None, operation_name: str = "Operation"):
        self.ctx = ctx
        self.operation_name = operation_name
        self.current = 0
        self.total = 100
        self.message = ""
        self._callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable[[int, int, str], None]):
        """Add a callback to be notified of progress updates."""
        self._callbacks.append(callback)
    
    async def update(self, current: int, total: int, message: str):
        """Update progress asynchronously."""
        self.current = current
        self.total = total
        self.message = message
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(current, total, message)
                else:
                    callback(current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Report to MCP context if available
        if self.ctx and hasattr(self.ctx, 'report_progress'):
            try:
                await self.ctx.report_progress(
                    progress=current,
                    total=total,
                    message=f"{self.operation_name}: {message}"
                )
            except Exception as e:
                logger.debug(f"MCP progress report failed: {e}")
    
    def update_sync(self, current: int, total: int, message: str):
        """Update progress synchronously (for non-async contexts)."""
        self.current = current
        self.total = total
        self.message = message
        
        # Notify callbacks (sync only)
        for callback in self._callbacks:
            try:
                if not asyncio.iscoroutinefunction(callback):
                    callback(current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def get_sync_callback(self) -> Callable[[int, int, str], None]:
        """Get a synchronous callback function for use in non-async code."""
        return self.update_sync