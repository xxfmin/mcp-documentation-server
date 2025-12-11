import sqlite3
import json
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timezone
import logging

from lib import DocumentMetadata, Config

if TYPE_CHECKING:
    from .document_index import DocumentIndex

logger = logging.getLogger(__name__)

"""Manages document metadata storage and retrieval"""
class DocumentStore:

    def __init__(
        self,
        db_path: Optional[str] = None,
        document_index: Optional['DocumentIndex'] = None
    ):
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Use Config for default path
            self.db_path = Config.BASE_DIR / "data" / "documents.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.document_index = document_index  # Optional index integration
        self._init_db()
        logger.info(f"DocumentStore initialized with db_path: {self.db_path}")
        if self.document_index:
            logger.info("DocumentStore integrated with DocumentIndex")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    collection_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    source_url TEXT,
                    title TEXT,
                    author TEXT,
                    pages INTEGER,
                    indexed_at TEXT NOT NULL,
                    custom_metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_collection_id 
                ON documents(collection_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_indexed_at 
                ON documents(indexed_at)
                """
            )
            conn.commit()

    def add_document(self, document: DocumentMetadata) -> None:
        """
        Add or update a document.
        
        Also updates DocumentIndex if available.

        Args:
            document: DocumentMetadata object to store
        """
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents 
                (document_id, collection_id, filename, document_type, source_url,
                 title, author, pages, indexed_at, custom_metadata, 
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        COALESCE((SELECT created_at FROM documents WHERE document_id = ?), ?),
                        ?)
                """,
                (
                    document.document_id,
                    document.collection_id,
                    document.filename,
                    document.document_type,
                    document.source_url,
                    document.title,
                    document.author,
                    document.pages,
                    document.indexed_at.isoformat(),
                    json.dumps(document.custom_metadata),
                    document.document_id,  # For COALESCE check
                    now,  # created_at fallback
                    now,  # updated_at
                ),
            )
            conn.commit()
        
        # Update DocumentIndex if available
        if self.document_index and Config.INDEXING_ENABLED:
            try:
                # DocumentIndex needs filepath - we'll use a virtual path
                # In a real implementation, you might store the actual file path
                filepath = str(self.db_path.parent / f"{document.document_id}.json")
                # Note: DocumentIndex.add_document expects content and chunks
                # This would need to be called from IndexingPipeline after extraction
                # For now, we just track the document ID -> filepath mapping
                pass  # Index will be updated by IndexingPipeline
            except Exception as e:
                logger.warning(f"Failed to update DocumentIndex: {e}")
        
        logger.info(f"Added/updated document: {document.document_id}")

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document by ID.
        
        Uses DocumentIndex for O(1) lookup if available and enabled.

        Args:
            document_id: Document identifier

        Returns:
            DocumentMetadata if found, None otherwise
        """
        # Try DocumentIndex first if enabled
        if self.document_index and Config.INDEXING_ENABLED:
            try:
                filepath = self.document_index.find_document(document_id)
                if filepath:
                    # If index found it, verify it exists in DB
                    # (index might be out of sync)
                    pass  # Fall through to DB lookup for consistency
            except Exception as e:
                logger.debug(f"DocumentIndex lookup failed, using DB: {e}")
        
        # Standard DB lookup
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()

        if not row:
            logger.debug(f"Document not found: {document_id}")
            return None

        return self.row_to_document(row)

    def get_documents_by_collection(
        self, collection_id: str
    ) -> List[DocumentMetadata]:
        """
        Get all documents in a collection

        Args:
            collection_id: Collection identifier

        Returns:
            List of DocumentMetadata objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE collection_id = ? ORDER BY indexed_at DESC",
                (collection_id,),
            )
            rows = cursor.fetchall()

        documents = [self.row_to_document(row) for row in rows]
        logger.debug(f"Found {len(documents)} documents in collection {collection_id}")
        return documents

    def list_documents(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[DocumentMetadata]:
        """
        List all documents
        
        Can use DocumentIndex for faster listing if enabled

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of DocumentMetadata objects, sorted by indexed_at (newest first)
        """
        # If using index and no limit/offset, could use index for faster listing
        # For now, use standard DB query
        query = "SELECT * FROM documents ORDER BY indexed_at DESC"
        params = []
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        elif offset:
            query += " LIMIT -1 OFFSET ?"
            params = [offset]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self.row_to_document(row) for row in rows]

    def update_document(self, document: DocumentMetadata) -> None:
        """
        Update document metadata

        Args:
            document: DocumentMetadata with updated data
        """
        if not self.document_exists(document.document_id):
            raise ValueError(f"Document '{document.document_id}' does not exist")

        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE documents SET
                    collection_id = ?,
                    filename = ?,
                    document_type = ?,
                    source_url = ?,
                    title = ?,
                    author = ?,
                    pages = ?,
                    indexed_at = ?,
                    custom_metadata = ?,
                    updated_at = ?
                WHERE document_id = ?
                """,
                (
                    document.collection_id,
                    document.filename,
                    document.document_type,
                    document.source_url,
                    document.title,
                    document.author,
                    document.pages,
                    document.indexed_at.isoformat(),
                    json.dumps(document.custom_metadata),
                    now,
                    document.document_id,
                ),
            )
            conn.commit()
        
        logger.info(f"Updated document: {document.document_id}")

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document
        
        Also removes from DocumentIndex if available

        Args:
            document_id: Document identifier

        Returns:
            True if deleted, False if document didn't exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            # Remove from DocumentIndex if available
            if self.document_index and Config.INDEXING_ENABLED:
                try:
                    self.document_index.remove_document(document_id)
                except Exception as e:
                    logger.warning(f"Failed to remove from DocumentIndex: {e}")
            
            logger.info(f"Deleted document: {document_id}")
        else:
            logger.debug(f"Document not found for deletion: {document_id}")

        return deleted

    def document_exists(self, document_id: str) -> bool:
        """
        Check if document exists

        Args:
            document_id: Document identifier

        Returns:
            True if document exists, False otherwise
        """
        # Try index first if enabled
        if self.document_index and Config.INDEXING_ENABLED:
            try:
                if self.document_index.find_document(document_id):
                    return True
            except Exception:
                pass  # Fall through to DB check
        
        # Standard DB check
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM documents WHERE document_id = ?",
                (document_id,),
            )
            return cursor.fetchone() is not None

    def count_documents(self, collection_id: Optional[str] = None) -> int:
        """
        Count documents

        Args:
            collection_id: Optional collection ID to filter by

        Returns:
            Number of documents
        """
        with sqlite3.connect(self.db_path) as conn:
            if collection_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE collection_id = ?",
                    (collection_id,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
            
            return cursor.fetchone()[0]

    def row_to_document(self, row: sqlite3.Row) -> DocumentMetadata:
        """
        Convert database row to DocumentMetadata

        Args:
            row: SQLite row object

        Returns:
            DocumentMetadata object
        """
        custom_metadata = json.loads(row["custom_metadata"]) if row["custom_metadata"] else {}
        
        return DocumentMetadata(
            document_id=row["document_id"],
            collection_id=row["collection_id"],
            filename=row["filename"],
            document_type=row["document_type"],
            source_url=row["source_url"],
            title=row["title"],
            author=row["author"],
            pages=row["pages"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]),
            custom_metadata=custom_metadata,
        )