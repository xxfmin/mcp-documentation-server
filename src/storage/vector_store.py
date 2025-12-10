from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import logging
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel
import pyarrow as pa

from lib import DocumentChunk, ChunkMetadata, Config
from core.embedding import Embedder

logger = logging.getLogger(__name__)

# LanceDB schema for chunk metadata
class ChunkMetadataLance(LanceModel):
    # Document references
    chunk_id: str
    collection_id: str
    document_id: str

    # Position metadata
    char_end: int
    char_start: int
    chunk_index: int
    heading_level: Optional[int] = None
    page_numbers: Optional[List[int]] = None

    # Structural context
    section_hierarchy: List[str] = []

    # Source info
    document_type: str
    filename: str
    indexed_at: str  # ISO format string
    source_url: Optional[str] = None
    total_chunks: int

# LanceDB schema for document chunks with embeddings
class ChunkVector(LanceModel):

    metadata: ChunkMetadataLance
    text: str  # Source field for embeddings

"""Manages vector storage for document chunks"""
class VectorStore:
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        default_embedding_model: Optional[str] = None,
    ):
        """
        Initialize VectorStore.

        Args:
            db_path: Path to LanceDB database directory
                    (uses Config.BASE_DIR/data/lancedb if None)
            embedder: Optional Embedder instance (creates new with cache if None)
            default_embedding_model: Default embedding model name
                                    (uses Config.EMBEDDING_MODEL if None)
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Use Config for default path
            self.db_path = Config.BASE_DIR / "data" / "lancedb"
        
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Use provided embedder or create one with cache from Config
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = Embedder(
                default_model=default_embedding_model or Config.EMBEDDING_MODEL
            )
            # Cache is automatically enabled in Embedder based on Config
        
        self.default_embedding_model = default_embedding_model or Config.EMBEDDING_MODEL
        self.db = lancedb.connect(str(self.db_path))
        self._tables = {}  # Cache for table references
        
        logger.info(f"VectorStore initialized with db_path: {self.db_path}")
        logger.info(f"Using embedding model: {self.default_embedding_model}")
        if Config.CACHE_ENABLED:
            logger.info(f"Embedding cache enabled (size: {Config.CACHE_SIZE})")

    def get_table_name(self, collection_id: Optional[str] = None) -> str:
        """
        Get table name for a collection.
        
        Args:
            collection_id: Optional collection ID (uses 'default' if None)
            
        Returns:
            Table name
        """
        if collection_id:
            return f"chunks_{collection_id}"
        return "chunks_default"

    def get_table_schema(self, embedding_model: Optional[str] = None) -> pa.Schema:
        """
        Create PyArrow schema for direct vector storage (FAST - pre-computed embeddings).
        
        This schema stores pre-computed vectors directly without auto-embedding,
        which is much faster for bulk inserts.
        
        Args:
            embedding_model: Optional model name for dimension lookup
            
        Returns:
            PyArrow schema for the table
        """
        dim = self.embedder.get_embedding_dimension(embedding_model)
        
        return pa.schema([
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("metadata", pa.struct([
                pa.field("chunk_id", pa.string()),
                pa.field("collection_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("char_end", pa.int64()),
                pa.field("char_start", pa.int64()),
                pa.field("chunk_index", pa.int64()),
                pa.field("heading_level", pa.int64(), nullable=True),
                pa.field("page_numbers", pa.list_(pa.int64()), nullable=True),
                pa.field("section_hierarchy", pa.list_(pa.string())),
                pa.field("document_type", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("indexed_at", pa.string()),
                pa.field("source_url", pa.string(), nullable=True),
                pa.field("total_chunks", pa.int64()),
            ])),
        ])

    def ensure_table(
        self,
        collection_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        mode: str = "create",
    ):
        """
        Ensure table exists for a collection.
        
        Args:
            collection_id: Collection ID (uses 'default' if None)
            embedding_model: Optional embedding model override
            mode: Table creation mode ('create' or 'overwrite')
                  Note: 'append' is not supported by LanceDB for table creation.
                  If table exists, it will be opened instead of created.
        """
        table_name = self.get_table_name(collection_id)
        
        # First, always try to open the table from the database
        try:
            table = self.db.open_table(table_name)
            self._tables[table_name] = table
            logger.debug(f"Opened existing table: {table_name}")
            return
        except Exception:
            # Table doesn't exist in database, will create below
            pass

        # Create PyArrow schema for direct vector storage (faster than auto-embed)
        schema = self.get_table_schema(embedding_model)
        
        # LanceDB only supports 'create' or 'overwrite' modes for create_table
        actual_mode = "create" if mode == "append" else mode
        
        # Create table with PyArrow schema (no data, just schema)
        try:
            table = self.db.create_table(table_name, schema=schema, mode=actual_mode)
            self._tables[table_name] = table
            logger.info(f"Created table: {table_name}")
        except Exception as e:
            # If creation failed because table already exists (race condition),
            # try to open it
            if "already exists" in str(e).lower():
                try:
                    table = self.db.open_table(table_name)
                    self._tables[table_name] = table
                    logger.debug(f"Table {table_name} already existed, opened it")
                    return
                except Exception as open_error:
                    logger.error(f"Failed to open existing table {table_name}: {open_error}")
                    raise
            logger.error(f"Failed to create table {table_name}: {e}")
            raise

    def get_table(self, collection_id: Optional[str] = None):
        """
        Get table for a collection.
        
        Args:
            collection_id: Collection ID
            
        Returns:
            LanceDB table object
        """
        table_name = self.get_table_name(collection_id)
        
        if table_name not in self._tables:
            try:
                table = self.db.open_table(table_name)
                self._tables[table_name] = table
            except Exception as e:
                logger.error(f"Table {table_name} does not exist: {e}")
                raise
        
        return self._tables[table_name]

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """
        Add chunks to the vector store with pre-computed embeddings.
        
        This method pre-computes embeddings in batches (with progress callbacks)
        before inserting into LanceDB. Uses cached Embedder for performance.

        Args:
            chunks: List of DocumentChunk objects
            collection_id: Collection ID
            embedding_model: Optional embedding model override
            progress_callback: Optional callback(current, total, message) for progress reporting
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Ensure table exists
        self.ensure_table(collection_id, embedding_model, mode="append")
        
        # Step 1: Extract texts for embedding
        texts = [chunk.text for chunk in chunks]
        
        # Step 2: Pre-compute embeddings with progress callback
        # Embedder will use cache if enabled (from Config)
        logger.info(f"Computing embeddings for {len(texts)} chunks...")
        if progress_callback:
            progress_callback(0, len(chunks), f"Starting embedding computation for {len(chunks)} chunks")
        
        embeddings = self.embedder.embed_batch(
            texts=texts,
            model_name=embedding_model,
            batch_size=32,  # Smaller batches for more frequent progress updates
            progress_callback=progress_callback
        )
        
        logger.info(f"Computed {len(embeddings)} embeddings")
        
        # Log cache stats if available
        if Config.CACHE_ENABLED:
            cache_stats = self.embedder.get_cache_stats()
            if cache_stats:
                logger.debug(
                    f"Embedding cache: {cache_stats['hits']} hits, "
                    f"{cache_stats['misses']} misses, "
                    f"hit rate: {cache_stats['hit_rate']:.2%}"
                )
        
        # Step 3: Prepare chunks with pre-computed vectors for LanceDB
        lance_chunks = []
        for i, chunk in enumerate(chunks):
            lance_metadata = {
                "chunk_id": chunk.chunk_id,
                "collection_id": chunk.metadata.collection_id,
                "document_id": chunk.metadata.document_id,
                "char_end": chunk.metadata.char_end,
                "char_start": chunk.metadata.char_start,
                "chunk_index": chunk.metadata.chunk_index,
                "heading_level": chunk.metadata.heading_level,
                "page_numbers": chunk.metadata.page_numbers,
                "section_hierarchy": chunk.metadata.section_hierarchy,
                "document_type": chunk.metadata.document_type,
                "filename": chunk.metadata.filename,
                "indexed_at": chunk.metadata.indexed_at.isoformat(),
                "source_url": chunk.metadata.source_url,
                "total_chunks": chunk.metadata.total_chunks,
            }
            
            lance_chunk = {
                "text": chunk.text,
                "vector": embeddings[i],  # Pre-computed embedding
                "metadata": lance_metadata,
            }
            lance_chunks.append(lance_chunk)

        # Step 4: Add to table (fast - no embedding computation)
        if progress_callback:
            progress_callback(len(chunks), len(chunks), "Storing chunks in database...")
        
        try:
            table = self.get_table(collection_id)
            table.add(lance_chunks)
            logger.info(f"Added {len(chunks)} chunks to collection {collection_id or 'default'}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise

    def search(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Search chunks using vector similarity.
        
        Args:
            query: Search query text
            collection_id: Optional collection ID to filter by
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"document_id": "doc_123"})
            
        Returns:
            List of DocumentChunk objects (with scores stored in a custom attribute)
        """
        try:
            table = self.get_table(collection_id)
            
            # Build search query
            search_query = table.search(query).limit(top_k)
            
            # Apply metadata filters if provided
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    if key == "document_id":
                        where_clauses.append(f"metadata.document_id = '{value}'")
                    elif key == "collection_id":
                        where_clauses.append(f"metadata.collection_id = '{value}'")
                    elif key == "document_type":
                        where_clauses.append(f"metadata.document_type = '{value}'")
                    # Add more filter types as needed
                
                if where_clauses:
                    where_expr = " AND ".join(where_clauses)
                    search_query = search_query.where(where_expr)
            
            # Execute search
            results = search_query.to_pandas()
            
            # Convert to DocumentChunk objects
            chunks = []
            for _, row in results.iterrows():
                chunk = self.row_to_chunk(row)
                if chunk:
                    # Store the similarity score from LanceDB's _distance column
                    # LanceDB returns distance (lower is better), convert to similarity
                    if '_distance' in row:
                        # Convert distance to similarity score (1 / (1 + distance))
                        distance = row['_distance']
                        # chunk.search_score = 1.0 / (1.0 + distance)
                    chunks.append(chunk)
            
            logger.debug(f"Search returned {len(chunks)} results for query: {query[:50]}")
            return chunks
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def delete_chunks_by_document(
        self,
        document_id: str,
        collection_id: Optional[str] = None,
    ) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            collection_id: Optional collection ID
            
        Returns:
            Number of chunks deleted
        """
        try:
            table = self.get_table(collection_id)
            table_name = self.get_table_name(collection_id)
            
            # Get all chunks
            all_data = table.to_pandas()
            
            if len(all_data) == 0:
                # Table is empty
                return 0
            
            # Filter out chunks for this document
            filtered_data = all_data[
                all_data["metadata"].apply(lambda x: x.get("document_id") != document_id)
            ]
            
            if len(filtered_data) == len(all_data):
                # No chunks to delete
                return 0
            
            deleted_count = len(all_data) - len(filtered_data)
            
            # Recreate table with filtered data
            if len(filtered_data) > 0:
                # Convert pandas DataFrame back to list of dicts with proper structure
                filtered_chunks = []
                for _, row in filtered_data.iterrows():
                    # Handle metadata - it might be a dict or a pandas Series
                    metadata = row["metadata"]
                    if not isinstance(metadata, dict):
                        if hasattr(metadata, 'to_dict'):
                            metadata = metadata.to_dict()
                        else:
                            metadata = dict(metadata)
                    
                    # Handle vector - ensure it's a list
                    vector = row.get("vector")
                    if vector is not None:
                        if hasattr(vector, 'tolist'):
                            vector = vector.tolist()
                        elif not isinstance(vector, list):
                            vector = list(vector)
                    
                    chunk_dict = {
                        "text": str(row["text"]),
                        "vector": vector,
                        "metadata": metadata,
                    }
                    filtered_chunks.append(chunk_dict)
                
                # Drop the existing table
                self.db.drop_table(table_name)
                if table_name in self._tables:
                    del self._tables[table_name]
                
                # Recreate table with filtered data
                # Use default embedding model for schema
                embedding_model = self.default_embedding_model
                
                # Create new table with schema
                schema = self.get_table_schema(embedding_model)
                new_table = self.db.create_table(table_name, schema=schema, mode="create")
                
                # Add filtered chunks back
                if filtered_chunks:
                    new_table.add(filtered_chunks)
                
                # Cache the new table
                self._tables[table_name] = new_table
            else:
                # Delete table if empty
                self.db.drop_table(table_name)
                if table_name in self._tables:
                    del self._tables[table_name]
            
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            raise

    def count_chunks(
        self,
        collection_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> int:
        """
        Count chunks in the store.
        
        Args:
            collection_id: Optional collection ID
            document_id: Optional document ID to filter by
            
        Returns:
            Number of chunks
        """
        try:
            table = self.get_table(collection_id)
            data = table.to_pandas()
            
            if document_id:
                filtered = data[
                    data["metadata"].apply(lambda x: x.get("document_id") == document_id)
                ]
                return len(filtered)
            
            return len(data)
        except Exception as e:
            logger.error(f"Failed to count chunks: {e}")
            return 0

    def row_to_chunk(self, row) -> Optional[DocumentChunk]:
        """
        Convert database row to DocumentChunk.
        
        Args:
            row: Pandas Series or dict from LanceDB
            
        Returns:
            DocumentChunk object or None
        """
        try:
            # Handle both Series and dict
            if hasattr(row, "to_dict"):
                row_dict = row.to_dict()
            else:
                row_dict = dict(row)
            
            metadata_dict = row_dict.get("metadata", {})
            if isinstance(metadata_dict, dict):
                # Convert metadata
                chunk_metadata = ChunkMetadata(
                    chunk_id=metadata_dict["chunk_id"],
                    document_id=metadata_dict["document_id"],
                    collection_id=metadata_dict["collection_id"],
                    chunk_index=metadata_dict["chunk_index"],
                    total_chunks=metadata_dict["total_chunks"],
                    char_start=metadata_dict["char_start"],
                    char_end=metadata_dict["char_end"],
                    page_numbers=metadata_dict.get("page_numbers"),
                    section_hierarchy=metadata_dict.get("section_hierarchy", []),
                    heading_level=metadata_dict.get("heading_level"),
                    filename=metadata_dict["filename"],
                    document_type=metadata_dict["document_type"],
                    source_url=metadata_dict.get("source_url"),
                    indexed_at=datetime.fromisoformat(metadata_dict["indexed_at"]),
                )
                
                # Get embedding if available
                embedding = None
                if "vector" in row_dict:
                    vector = row_dict["vector"]
                    if hasattr(vector, "tolist"):
                        embedding = vector.tolist()
                    elif isinstance(vector, list):
                        embedding = vector
                
                # Create DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=chunk_metadata.chunk_id,
                    text=row_dict.get("text", ""),
                    metadata=chunk_metadata,
                    embedding=embedding,
                )
                
                return chunk
        except Exception as e:
            logger.error(f"Failed to convert row to chunk: {e}")
            return None