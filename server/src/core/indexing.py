from typing import List, Optional, Dict, Any, Callable
import logging
from pathlib import Path

from .extraction import DocumentExtractor
from .chunking import Chunker
from .embedding import Embedder
from lib import Config, generate_document_id

from storage import (
    CollectionManager,
    DocumentStore,
    VectorStore,
    DocumentIndex,
)

logger = logging.getLogger(__name__)

"""Indexing pipeline for end-to-end documentation processing"""
class IndexingResult:

    def __init__(
        self,
        document_id: str,
        collection_id: str,
        chunk_count: int,
        status: str,
        error: Optional[str] = None,
    ):
        self.document_id = document_id
        self.collection_id = collection_id
        self.chunk_count = chunk_count
        self.status = status  # "indexed", "failed", "skipped"
        self.error = error


class IndexingPipeline:
    """
    Orchestrates end-to-end document indexing.
    
    Updated to integrate:
    - Config for defaults and feature flags
    - DocumentIndex for O(1) lookups
    - Streaming support for large files
    - Parallel chunking when enabled
    """

    def __init__(
        self,
        collection_manager: Optional[CollectionManager] = None,
        document_store: Optional[DocumentStore] = None,
        vector_store: Optional[VectorStore] = None,
        document_index: Optional[DocumentIndex] = None,
        extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
    ):
        """
        Initialize IndexingPipeline.

        Args:
            collection_manager: CollectionManager instance (creates new if None)
            document_store: DocumentStore instance (creates new if None)
            vector_store: VectorStore instance (creates new if None)
            document_index: Optional DocumentIndex instance (creates new if enabled)
            extractor: DocumentExtractor instance (creates new if None)
            chunker: Chunker instance (creates new if None)
            embedder: Embedder instance (creates new if None)
        """
        self.collection_manager = collection_manager or CollectionManager()
        self.document_store = document_store or DocumentStore()
        self.vector_store = vector_store or VectorStore()
        
        # Initialize DocumentIndex if enabled
        if Config.INDEXING_ENABLED and document_index is None:
            try:
                self.document_index = DocumentIndex(
                    data_dir=Config.BASE_DIR / "data"
                )
                self.document_index.initialize()
                # Link DocumentIndex to DocumentStore
                self.document_store.document_index = self.document_index
                logger.info("DocumentIndex enabled and initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DocumentIndex: {e}")
                self.document_index = None
        else:
            self.document_index = document_index
        
        self.extractor = extractor or DocumentExtractor()
        self.chunker = chunker or Chunker()
        self.embedder = embedder or Embedder()

        logger.info("IndexingPipeline initialized")
        if Config.INDEXING_ENABLED and self.document_index:
            logger.info(f"DocumentIndex stats: {self.document_index.get_stats()}")

    def index_document(
        self,
        source: str,
        collection_id: str,
        document_type: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        embedding_model: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> IndexingResult:
        """
        Index a single document.
        
        Enhanced with:
        - Streaming support for large files
        - Parallel chunking when enabled
        - DocumentIndex integration
        - Config-based defaults

        Args:
            source: URL or file path to document
            collection_id: Target collection ID
            document_type: Optional document type override
            custom_metadata: Optional custom metadata
            force_reindex: If True, re-index even if document exists
            embedding_model: Optional embedding model override
            progress_callback: Optional callback(current, total, message) for progress reporting

        Returns:
            IndexingResult with status and metadata
        """
        document_id = ""
        
        try:
            # Check if collection exists
            collection = self.collection_manager.get_collection(collection_id)
            if not collection:
                return IndexingResult(
                    document_id="",
                    collection_id=collection_id,
                    chunk_count=0,
                    status="failed",
                    error=f"Collection '{collection_id}' does not exist",
                )

            # Generate document ID
            document_id = generate_document_id(source)

            # Check for duplicate content using DocumentIndex if enabled
            if Config.INDEXING_ENABLED and self.document_index and not force_reindex:
                # We can't check content hash until after extraction
                # But we can check if document exists
                if self.document_store.document_exists(document_id):
                    logger.info(f"Document {document_id} already exists, skipping")
                    existing_doc = self.document_store.get_document(document_id)
                    return IndexingResult(
                        document_id=document_id,
                        collection_id=collection_id,
                        chunk_count=0,
                        status="skipped",
                        error="Document already indexed",
                    )

            # Step 1: Extract document (with streaming support if enabled)
            logger.info(f"Extracting document: {source}")
            if progress_callback:
                progress_callback(1, 100, "Extracting document...")
            
            # Check if streaming is needed
            source_path = Path(source) if not source.startswith("http") else None
            if source_path and source_path.exists() and Config.STREAMING_ENABLED:
                file_size = source_path.stat().st_size
                if file_size > Config.STREAM_FILE_SIZE_LIMIT:
                    logger.info(f"Large file detected ({file_size} bytes), using streaming extraction")
            
            try:
                doc, doc_metadata = self.extractor.extract(
                    source=source,
                    collection_id=collection_id,
                    custom_metadata=custom_metadata,
                )
            except Exception as e:
                logger.error(f"Extraction failed for {source}: {e}", exc_info=True)
                return IndexingResult(
                    document_id=document_id,
                    collection_id=collection_id,
                    chunk_count=0,
                    status="failed",
                    error=f"Extraction failed: {str(e)}",
                )

            # Check for duplicate content using DocumentIndex
            if Config.INDEXING_ENABLED and self.document_index and not force_reindex:
                # Get document content for hashing
                doc_content = getattr(doc, 'text', '') or str(doc)
                duplicate_id = self.document_index.find_duplicate_content(doc_content)
                if duplicate_id and duplicate_id != document_id:
                    logger.warning(f"Duplicate content detected, existing document: {duplicate_id}")
                    # Optionally return existing document or continue with new

            # Override document type if provided
            if document_type:
                doc_metadata.document_type = document_type

            # Step 2: Chunk document (with parallel processing if enabled)
            logger.info(f"Chunking document: {doc_metadata.document_id}")
            if progress_callback:
                progress_callback(10, 100, "Chunking document...")
            
            try:
                # Chunker will use parallel processing if enabled and document is large
                chunks = self.chunker.chunk_document(doc, doc_metadata)
            except Exception as e:
                logger.error(f"Chunking failed for {doc_metadata.document_id}: {e}", exc_info=True)
                return IndexingResult(
                    document_id=doc_metadata.document_id,
                    collection_id=collection_id,
                    chunk_count=0,
                    status="failed",
                    error=f"Chunking failed: {str(e)}",
                )

            if not chunks:
                logger.warning(f"No chunks generated for document {doc_metadata.document_id}")
                return IndexingResult(
                    document_id=doc_metadata.document_id,
                    collection_id=collection_id,
                    chunk_count=0,
                    status="failed",
                    error="No chunks generated",
                )

            # Step 3: Store document metadata
            logger.info(f"Storing document metadata: {doc_metadata.document_id}")
            if progress_callback:
                progress_callback(15, 100, "Storing document metadata...")
            try:
                self.document_store.add_document(doc_metadata)
            except Exception as e:
                logger.error(f"Failed to store document metadata: {e}", exc_info=True)
                return IndexingResult(
                    document_id=doc_metadata.document_id,
                    collection_id=collection_id,
                    chunk_count=0,
                    status="failed",
                    error=f"Metadata storage failed: {str(e)}",
                )

            # Step 4: Update DocumentIndex
            if Config.INDEXING_ENABLED and self.document_index:
                try:
                    # Get document content for indexing
                    doc_content = getattr(doc, 'text', '') or str(doc)
                    filepath = str(Config.BASE_DIR / "data" / f"{document_id}.json")
                    self.document_index.add_document(
                        document_id=document_id,
                        filepath=filepath,
                        content=doc_content,
                        chunks=chunks,
                    )
                    logger.debug(f"Added document {document_id} to DocumentIndex")
                except Exception as e:
                    logger.warning(f"Failed to update DocumentIndex: {e}")

            # Step 5: Store chunks with embeddings (this is the slow part)
            logger.info(
                f"Storing {len(chunks)} chunks with embeddings: {doc_metadata.document_id}"
            )
            if progress_callback:
                progress_callback(20, 100, f"Computing embeddings for {len(chunks)} chunks...")
            
            # Create a wrapper callback that scales embedding progress to 20-95%
            def embedding_progress_callback(current: int, total: int, message: str):
                if progress_callback:
                    # Scale from 20% to 95% based on embedding progress
                    scaled_progress = 20 + int((current / max(total, 1)) * 75)
                    progress_callback(scaled_progress, 100, message)
            
            try:
                embedding_model = embedding_model or collection.embedding_model
                chunks_added = self.vector_store.add_chunks(
                    chunks=chunks,
                    collection_id=collection_id,
                    embedding_model=embedding_model,
                    progress_callback=embedding_progress_callback,
                )
            except Exception as e:
                logger.error(f"Failed to store chunks: {e}", exc_info=True)
                # Clean up: remove document metadata and index entry since chunks failed
                try:
                    self.document_store.delete_document(doc_metadata.document_id)
                    if Config.INDEXING_ENABLED and self.document_index:
                        self.document_index.remove_document(doc_metadata.document_id)
                except Exception:
                    pass
                return IndexingResult(
                    document_id=doc_metadata.document_id,
                    collection_id=collection_id,
                    chunk_count=0,
                    status="failed",
                    error=f"Chunk storage failed: {str(e)}",
                )

            # Step 6: Update collection counts
            if progress_callback:
                progress_callback(98, 100, "Updating collection counts...")
            
            if not force_reindex or not self.document_store.document_exists(document_id):
                # Only increment if this is a new document
                self.collection_manager.increment_document_count(collection_id, 1)
            self.collection_manager.increment_chunk_count(collection_id, chunks_added)

            if progress_callback:
                progress_callback(100, 100, f"Successfully indexed {chunks_added} chunks")

            logger.info(
                f"Successfully indexed document {doc_metadata.document_id} "
                f"with {chunks_added} chunks"
            )

            return IndexingResult(
                document_id=doc_metadata.document_id,
                collection_id=collection_id,
                chunk_count=chunks_added,
                status="indexed",
            )

        except Exception as e:
            logger.error(f"Failed to index document {source}: {e}", exc_info=True)
            return IndexingResult(
                document_id=document_id,
                collection_id=collection_id,
                chunk_count=0,
                status="failed",
                error=str(e),
            )

    def index_documents_batch(
        self,
        sources: List[str],
        collection_id: str,
        document_type: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        embedding_model: Optional[str] = None,
    ) -> List[IndexingResult]:
        """
        Index multiple documents.
        
        Processes documents sequentially (could be parallelized in future).

        Args:
            sources: List of URLs or file paths
            collection_id: Target collection ID
            document_type: Optional document type override
            custom_metadata: Optional custom metadata (applied to all)
            force_reindex: If True, re-index even if documents exist
            embedding_model: Optional embedding model override

        Returns:
            List of IndexingResult objects
        """
        results = []

        for source in sources:
            result = self.index_document(
                source=source,
                collection_id=collection_id,
                document_type=document_type,
                custom_metadata=custom_metadata,
                force_reindex=force_reindex,
                embedding_model=embedding_model,
            )
            results.append(result)

        return results

    def delete_document(
        self,
        document_id: str,
        collection_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a document and all its chunks.
        
        Also removes from DocumentIndex if enabled.

        Args:
            document_id: Document ID to delete
            collection_id: Optional collection ID (auto-detected if None)

        Returns:
            True if deleted, False otherwise
        """
        try:
            # Get document to find collection_id if not provided
            if not collection_id:
                doc = self.document_store.get_document(document_id)
                if not doc:
                    logger.warning(f"Document {document_id} not found")
                    return False
                collection_id = doc.collection_id

            # Delete chunks from vector store
            chunks_deleted = self.vector_store.delete_chunks_by_document(
                document_id=document_id,
                collection_id=collection_id,
            )

            # Delete document metadata
            deleted = self.document_store.delete_document(document_id)
            
            # Remove from DocumentIndex if enabled
            if Config.INDEXING_ENABLED and self.document_index:
                try:
                    self.document_index.remove_document(document_id)
                except Exception as e:
                    logger.warning(f"Failed to remove from DocumentIndex: {e}")

            if deleted:
                # Update collection counts
                self.collection_manager.increment_document_count(collection_id, -1)
                self.collection_manager.increment_chunk_count(collection_id, -chunks_deleted)

                logger.info(
                    f"Deleted document {document_id} and {chunks_deleted} chunks"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}", exc_info=True)
            return False

    def reindex_document(
        self,
        source: str,
        collection_id: str,
        document_type: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[str] = None,
    ) -> IndexingResult:
        """
        Re-index a document (delete old, index new).

        Args:
            source: URL or file path to document
            collection_id: Target collection ID
            document_type: Optional document type override
            custom_metadata: Optional custom metadata
            embedding_model: Optional embedding model override

        Returns:
            IndexingResult with status and metadata
        """
        # Generate document ID
        document_id = generate_document_id(source)

        # Delete existing document
        if self.document_store.document_exists(document_id):
            logger.info(f"Deleting existing document {document_id} before re-indexing")
            self.delete_document(document_id, collection_id)

        # Index document
        return self.index_document(
            source=source,
            collection_id=collection_id,
            document_type=document_type,
            custom_metadata=custom_metadata,
            force_reindex=True,
            embedding_model=embedding_model,
        )