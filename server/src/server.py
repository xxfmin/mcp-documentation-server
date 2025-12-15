import logging
import os
from typing import List, Optional, Any, Literal
from fastmcp import FastMCP, Context
from dotenv import load_dotenv
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

from core import DocumentManager
from lib import (
    Config,
    MCPError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    CollectionNotEmptyError,
    CollectionAlreadyExistsError,
    format_error_response,
)

logger = logging.getLogger(__name__)

# Configure logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

mcp = FastMCP(name="documentation-server", version="1.0.0")
document_manager: Optional[DocumentManager] = None

def get_document_manager() -> DocumentManager:
    global document_manager
    if document_manager is None:
        logger.info("Initializing Document Manager...")
        document_manager = DocumentManager(
            base_dir=Config.BASE_DIR,
            embedding_model=Config.EMBEDDING_MODEL
        )
        logger.info("DocumentManager initialized")
    return document_manager

# Collection Management
@mcp.tool()
def create_collection(
    name: str,
    description: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """
    Create a new document collection.
    
    Args:
        name: Collection name (e.g., "design-docs", "api-specs")
        description: Human-readable description
        embedding_model: Override default embedding model
        chunk_size: Max chunk tokens (default: 512)
        chunk_overlap: Overlap tokens (default: 50)
        
    Returns:
        Dictionary with collection details on success, or error details on failure
    """
    try:
        return get_document_manager().create_collection(
            collection_id=name,
            description=description,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as e:
        # Collection already exists
        return CollectionAlreadyExistsError(name).to_dict()
    except Exception as e:
        logger.error(f"Failed to create collection: {e}", exc_info=True)
        return format_error_response(e)


@mcp.tool()
def list_collections() -> dict:
    """
    List all document collections.
    
    Returns:
        Dictionary containing a list of all collections with their metadata:
        - id: Collection identifier
        - description: Human-readable description
        - document_count: Number of documents in the collection
        - chunk_count: Total number of chunks in the collection
        - last_updated: ISO timestamp of last update
    """
    return {"collections": get_document_manager().list_collections()}


@mcp.tool()
def get_collection(name: str) -> dict:
    """
    Get detailed information about a specific collection.
    
    Args:
        name: Collection name/ID
        
    Returns:
        Dictionary with collection details:
        - id: Collection identifier
        - description: Human-readable description
        - embedding_model: Model used for embeddings
        - chunk_size: Maximum chunk size in tokens
        - chunk_overlap: Overlap between chunks in tokens
        - document_count: Number of documents
        - chunk_count: Total number of chunks
        - created_at: ISO timestamp of creation
        - last_updated: ISO timestamp of last update
    """
    collection = get_document_manager().get_collection(name)
    if not collection:
        return CollectionNotFoundError(
            name,
            [c["id"] for c in get_document_manager().list_collections()]
        ).to_dict()
    return collection


@mcp.tool()
def delete_collection(
    name: str,
    force: bool = False,
) -> dict:
    """
    Delete a collection and all its documents.
    
    Args:
        name: Collection name/ID to delete
        force: If True, delete even if collection contains documents.
               If False (default), will return error if collection is not empty.
               
    Returns:
        Dictionary with deletion result:
        - collection_id: The collection that was deleted
        - status: "deleted" on success
        - documents_deleted: Number of documents removed
        - chunks_deleted: Number of chunks removed
        
        On error, returns actionable error details with suggestions.
    """
    try:
        return get_document_manager().delete_collection(name, force=force)
    except CollectionNotFoundError as e:
        return e.to_dict()
    except CollectionNotEmptyError as e:
        return e.to_dict()
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}", exc_info=True)
        return format_error_response(e)

# Document Management
@mcp.tool()
async def index_document(
    collection_id: str,
    source: str,
    document_type: Optional[str] = None,
    metadata: Optional[dict] = None,
    force_reindex: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """
    Index a document from a URL or file path.
    
    The document will be extracted, chunked, embedded, and stored in the vector database.
    Supports PDF, HTML, Markdown, and DOCX formats.
    
    Args:
        collection_id: Collection ID where the document will be indexed
        source: URL or file path to the document (e.g., "https://example.com/doc.pdf" or "/path/to/file.pdf")
        document_type: Optional document type override (e.g., "pdf", "html", "markdown", "docx")
                       If not provided, will be auto-detected from source
        metadata: Optional dictionary of custom metadata to attach to the document
        force_reindex: If True, re-index even if document already exists (default: False)
        ctx: Optional MCP context for progress reporting (internal use)
    
    Returns:
        Dictionary with indexing result:
        - document_id: Unique document identifier
        - collection_id: Collection where document was indexed
        - source: Source URL or path
        - status: "indexed", "failed", or "skipped"
        - chunk_count: Number of chunks created
        - pages: Number of pages (if applicable)
        - indexed_at: ISO timestamp of indexing
        - metadata: Document metadata (filename, title, document_type)
    """
    manager = get_document_manager()

    # Progress callback for MCP
    progress_queue = []
    def progress_callback(current: int, total: int, message: str):
        progress_queue.append((current, total, message))
        if ctx: 
            # Would need async handling for MCP progress
            pass
    
    return manager.index_document_from_source(
        source=source,
        collection_id=collection_id,
        document_type=document_type,
        metadata=metadata,
        force_reindex=force_reindex,
        progress_callback=progress_callback,
    )

@mcp.tool()
def list_documents(collection_id: Optional[str] = None) -> dict:
    """
    List all documents in a collection or across all collections.
    
    Args:
        collection_id: Optional collection ID to filter documents.
                      If None, returns documents from all collections.
    
    Returns:
        Dictionary containing a list of documents with their metadata:
        - id: Document identifier
        - title: Document title
        - collection_id: Collection where document is stored
        - created_at: ISO timestamp of when document was indexed
        - metadata: Custom metadata dictionary
    """
    return {"documents": get_document_manager().get_all_documents(collection_id)}

@mcp.tool()
def get_document(document_id: str) -> dict:
    """
    Get detailed information about a specific document.
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        Dictionary with document details:
        - id: Document identifier
        - title: Document title
        - collection_id: Collection where document is stored
        - metadata: Custom metadata dictionary
        - created_at: ISO timestamp of when document was indexed
        - chunks_count: Total number of chunks in the document
        
        If document not found, returns actionable error with suggestions.
    """
    doc = get_document_manager().get_document(document_id)
    if not doc:
        return DocumentNotFoundError(document_id).to_dict()
    return doc


@mcp.tool()
def delete_document(document_id: str) -> dict:
    """
    Delete a document and all its associated chunks from the system.
    
    This operation removes:
    - Document metadata from the database
    - All chunks from the vector store
    - Document from any collections
    
    Args:
        document_id: Unique document identifier to delete
    
    Returns:
        Dictionary with deletion result:
        - document_id: The document ID that was requested for deletion
        - status: "deleted" if successful
        
        If document not found, returns actionable error with suggestions.
    """
    manager = get_document_manager()
    success = manager.delete_document(document_id)
    
    if success:
        return {
            "document_id": document_id,
            "status": "deleted",
        }
    else:
        return DocumentNotFoundError(document_id).to_dict()


@mcp.tool()
def update_document_metadata(
    document_id: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Update document metadata without re-indexing.
    
    Use this to change the title or add/update custom metadata for a document.
    The document content and chunks remain unchanged.
    
    Args:
        document_id: Unique document identifier
        title: New title for the document (optional)
        metadata: Dictionary of custom metadata to merge with existing metadata (optional).
                 Existing keys will be updated, new keys will be added.
    
    Returns:
        Dictionary with update result:
        - document_id: Document identifier
        - status: "updated" on success
        - title: Updated title
        - custom_metadata: Updated metadata dictionary
        - updated_at: ISO timestamp of update
        
        If document not found, returns actionable error with suggestions.
    """
    try:
        return get_document_manager().update_document_metadata(
            document_id=document_id,
            title=title,
            custom_metadata=metadata,
        )
    except DocumentNotFoundError as e:
        return e.to_dict()
    except Exception as e:
        logger.error(f"Failed to update document metadata: {e}", exc_info=True)
        return format_error_response(e)

# Search
@mcp.tool()
def search_documents(
    query: str,
    document_id: Optional[str] = None,
    collection_ids: Optional[List[str]] = None,
    top_k: int = 10,
    include_context: bool = False,
) -> dict:
    """
    Search documents using semantic similarity.
    
    Performs vector similarity search across document chunks. Results are ranked
    by relevance score (higher is better).
    
    Args:
        query: Natural language search query (e.g., "How to authenticate users?")
        document_id: Optional document ID to search within a specific document.
                    If provided, collection_ids is ignored.
        collection_ids: Optional list of collection IDs to search.
                       If None, searches across all collections.
        top_k: Maximum number of results to return (default: 10)
        include_context: If True, includes adjacent chunks as context for each result.
                       Useful for getting surrounding text around matches.
    
    Returns:
        Dictionary containing:
        - query: The original search query
        - results: List of search results, each containing:
          - chunk_id: Unique chunk identifier
          - document_id: Document containing the chunk
          - collection_id: Collection containing the document
          - chunk_index: Position of chunk within document
          - score: Similarity score (0.0 to 1.0, higher is better)
          - text: Chunk text content
          - metadata: Chunk metadata (filename, page_numbers, section, headings)
          - context: Optional context object with "before" and "after" text
        - total_results: Total number of results returned
    """
    results = get_document_manager().search_documents(
        query=query,
        document_id=document_id,
        collection_ids=collection_ids,
        top_k=top_k,
        include_context=include_context,
    )
    return {
        "query": query,
        "results": results,
        "total_results": len(results),
    }


@mcp.tool()
def hybrid_search(
    query: str,
    document_id: Optional[str] = None,
    collection_ids: Optional[List[str]] = None,
    top_k: int = 10,
    include_context: bool = False,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    rerank: bool = False,
    rerank_method: str = "combined",
) -> dict:
    """
    Search documents using hybrid search (vector similarity + keyword matching).
    
    Combines semantic search with keyword matching for improved precision.
    This is especially useful when you need both conceptual relevance AND
    specific term matching.
    
    Args:
        query: Natural language search query
        document_id: Optional document ID to search within a specific document
        collection_ids: Optional list of collection IDs to search
        top_k: Maximum number of results to return (default: 10)
        include_context: If True, includes adjacent chunks as context
        vector_weight: Weight for semantic similarity (0.0 to 1.0, default: 0.7)
        keyword_weight: Weight for keyword matching (0.0 to 1.0, default: 0.3)
        rerank: If True, apply additional reranking to improve results
        rerank_method: Reranking strategy:
                      - "keyword_boost": Boost exact keyword matches
                      - "length_penalty": Penalize very short/long chunks
                      - "position_boost": Favor chunks from document start
                      - "combined": Apply all strategies (default)
    
    Returns:
        Dictionary containing:
        - query: The original search query
        - search_type: "hybrid" to indicate search type used
        - weights: Object showing vector_weight and keyword_weight used
        - reranked: Boolean indicating if reranking was applied
        - results: List of search results (same format as search_documents)
        - total_results: Total number of results returned
    """
    results = get_document_manager().hybrid_search(
        query=query,
        document_id=document_id,
        collection_ids=collection_ids,
        top_k=top_k,
        include_context=include_context,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        rerank=rerank,
        rerank_method=rerank_method,
    )
    return {
        "query": query,
        "search_type": "hybrid",
        "weights": {
            "vector": vector_weight,
            "keyword": keyword_weight,
        },
        "reranked": rerank,
        "results": results,
        "total_results": len(results),
    }


@mcp.tool()
def get_context_window(
    document_id: str,
    chunk_index: int,
    before: int = 1,
    after: int = 1,
) -> dict:
    """
    Get a context window of chunks around a specific chunk in a document.
    
    Useful for retrieving surrounding text when you have a specific chunk index
    from a search result and want to see more context.
    
    Args:
        document_id: Document identifier
        chunk_index: Index of the center chunk (0-based)
        before: Number of chunks to include before the center chunk (default: 1)
        after: Number of chunks to include after the center chunk (default: 1)
    
    Returns:
        Dictionary containing:
        - window: List of chunks in the context window, each with:
          - chunk_index: Position of chunk within document
          - text: Chunk text content
        - center: The center chunk index
        - total_chunks: Total number of chunks in the document
        
    Raises:
        ValueError: If document or chunk index is not found
    """
    return get_document_manager().get_context_window(
        document_id=document_id,
        chunk_index=chunk_index,
        before=before,
        after=after,
    )

# File Upload
@mcp.tool()
def get_uploads_path() -> str:
    """
    Get the absolute path to the uploads folder.
    
    The uploads folder is where files can be placed for batch processing.
    Files placed here can be processed using the process_uploads tool.
    
    Returns:
        String containing the absolute path to the uploads folder
    """
    return f"Uploads folder: {get_document_manager().get_uploads_dir()}"


@mcp.tool()
def process_uploads() -> dict:
    """
    Process all files in the uploads folder.
    
    Scans the uploads directory and indexes all supported files found.
    Supported formats: PDF, Markdown (.md), and plain text (.txt).
    
    Returns:
        Dictionary with processing results:
        - processed: Number of files successfully processed
        - errors: List of error messages for files that failed to process
    """
    return get_document_manager().process_uploads_folder()


@mcp.tool()
def list_uploads_files() -> dict:
    """
    List all files in the uploads folder.
    
    Returns information about each file including whether it's a supported format
    that can be processed.
    
    Returns:
        Dictionary containing a list of files, each with:
        - name: Filename
        - size: File size in bytes
        - modified: Last modification timestamp
        - supported: Boolean indicating if file format is supported for indexing
    """
    return {"files": get_document_manager().list_uploads_files()}


# System
@mcp.tool()
def get_system_stats() -> dict:
    """
    Get comprehensive system statistics and configuration.
    
    Provides an overview of the documentation server's current state,
    including storage usage, document counts, and configuration settings.
    
    Returns:
        Dictionary containing:
        - collections: Object with total count and per-collection stats
        - documents: Object with total document count
        - chunks: Object with total chunk count
        - storage: Object with storage usage details:
          - data_directory: Path to data directory
          - total_size_bytes: Total storage used
          - total_size_formatted: Human-readable total size
          - vector_store_bytes: Vector database size
          - document_db_bytes: Document metadata database size
        - embedding_cache: Cache statistics if enabled
        - document_index: Index statistics if enabled
        - config: Current configuration settings:
          - embedding_model: Model used for embeddings
          - default_chunk_size: Default chunk size in tokens
          - default_chunk_overlap: Default overlap in tokens
          - cache_enabled: Whether embedding cache is enabled
          - parallel_enabled: Whether parallel processing is enabled
    """
    return get_document_manager().get_system_stats()


@mcp.tool()
def get_server_info() -> dict:
    """
    Get basic server information and available capabilities.
    
    Returns:
        Dictionary containing:
        - name: Server name
        - version: Server version
        - capabilities: List of available features
        - tools: List of available tool names
    """
    return {
        "name": "documentation-server",
        "version": "1.0.0",
        "capabilities": [
            "document_indexing",
            "semantic_search",
            "hybrid_search",
            "reranking",
            "collections",
            "metadata_management",
            "context_windows",
            "file_uploads",
        ],
        "tools": [
            # Collection Management
            "create_collection",
            "list_collections",
            "get_collection",
            "delete_collection",
            # Document Management
            "index_document",
            "list_documents",
            "get_document",
            "delete_document",
            "update_document_metadata",
            # Search
            "search_documents",
            "hybrid_search",
            "get_context_window",
            # File Upload
            "get_uploads_path",
            "process_uploads",
            "list_uploads_files",
            # System
            "get_system_stats",
            "get_server_info",
        ],
    }


def get_cors_middleware() -> list[Middleware]:
    """
    Create CORS middleware for SSE transport.
    
    Returns a list of Starlette middleware configured for CORS.
    """
    origins_str = Config.CORS_ORIGINS
    if origins_str == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [o.strip() for o in origins_str.split(",") if o.strip()]
    
    return [
        Middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]


def run_server():
    """
    Run the MCP server with the configured transport.
    
    Transport is configured via MCP_TRANSPORT environment variable:
    - "stdio": Standard input/output (default, for CLI/subprocess usage)
    - "sse": Server-Sent Events over HTTP (for web clients)
    - "streamable-http": Streamable HTTP (alternative to SSE)
    
    For SSE/HTTP transports, also configure:
    - MCP_HOST: Host to bind to (default: 127.0.0.1)
    - MCP_PORT: Port to bind to (default: 8000)
    - MCP_CORS_ORIGINS: Comma-separated allowed origins (default: http://localhost:3000)
    """
    transport = Config.TRANSPORT.lower()
    
    if transport == "stdio":
        logger.info("Starting MCP server with stdio transport")
        mcp.run(transport="stdio")
    
    elif transport in ("sse", "streamable-http"):
        host = Config.HOST
        port = Config.PORT
        
        logger.info(f"Starting MCP server with {transport} transport on {host}:{port}")
        logger.info(f"CORS origins: {Config.CORS_ORIGINS}")
        
        # Run with HTTP transport and CORS middleware
        mcp.run(
            transport=transport,  # type: ignore
            host=host,
            port=port,
            middleware=get_cors_middleware(),
        )
    
    else:
        raise ValueError(
            f"Unknown transport: {transport}. "
            f"Valid options are: stdio, sse, streamable-http"
        )


if __name__ == "__main__":
    run_server()