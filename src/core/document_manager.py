from typing import Callable, Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
import logging
import tempfile
import uuid

from lib import Config
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
            self.search_engine = SearchEngine(
                vector_store=self.vector_store,
                document_store=self.document_store,
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