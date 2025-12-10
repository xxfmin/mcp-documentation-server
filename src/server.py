import logging
from typing import List, Optional
from fastmcp import FastMCP, Context
from dotenv import load_dotenv

from core import DocumentManager
from lib import Config

mcp = FastMCP(name="Documentation Server")

load_dotenv()
logger = logging.getLogger(__name__)

mcp = FastMCP(name = "documentation-server", version="1.0.0")
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
    """
    return get_document_manager().create_collection(
        collection_id=name,
        description=description,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

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
        
        If document not found, returns:
        - error: Error message
    """
    doc = get_document_manager().get_document(document_id)
    if not doc:
        return {"error": f"Document {document_id} not found"}
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
        - status: "deleted" if successful, "not_found" if document doesn't exist
        - message: Optional helpful message (if document not found, suggests using list_documents)
    """
    manager = get_document_manager()
    success = manager.delete_document(document_id)
    
    if success:
        return {
            "document_id": document_id,
            "status": "deleted",
        }
    else:
        # Provide helpful message when document not found
        return {
            "document_id": document_id,
            "status": "not_found",
            "message": f"Document '{document_id}' was not found. Use 'list_documents' to see available documents.",
        }

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

if __name__ == "__main__":
    mcp.run()