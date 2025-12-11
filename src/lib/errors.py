"""Custom error classes with actionable error messages for the MCP server."""

from typing import Optional, List, Dict, Any


class MCPError(Exception):
    """Base error class for MCP server errors with actionable suggestions."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        suggestions: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.suggestions = suggestions or []
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        result = {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.suggestions:
            result["suggestions"] = self.suggestions
        if self.details:
            result["details"] = self.details
        return result


class CollectionNotFoundError(MCPError):
    """Raised when a collection is not found."""
    
    def __init__(self, collection_id: str, available_collections: Optional[List[str]] = None):
        suggestions = [
            f"Create the collection first using: create_collection(name='{collection_id}')",
            "Use list_collections() to see available collections",
        ]
        if available_collections:
            suggestions.append(f"Available collections: {', '.join(available_collections)}")
        
        super().__init__(
            message=f"Collection '{collection_id}' not found.",
            error_code="COLLECTION_NOT_FOUND",
            suggestions=suggestions,
            details={"collection_id": collection_id},
        )


class DocumentNotFoundError(MCPError):
    """Raised when a document is not found."""
    
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document '{document_id}' not found.",
            error_code="DOCUMENT_NOT_FOUND",
            suggestions=[
                "Use list_documents() to see available documents",
                "Check if the document_id is correct",
                "The document may have been deleted",
            ],
            details={"document_id": document_id},
        )


class CollectionAlreadyExistsError(MCPError):
    """Raised when trying to create a collection that already exists."""
    
    def __init__(self, collection_id: str):
        super().__init__(
            message=f"Collection '{collection_id}' already exists.",
            error_code="COLLECTION_EXISTS",
            suggestions=[
                f"Use a different name for your collection",
                f"Delete the existing collection first: delete_collection(name='{collection_id}')",
            ],
            details={"collection_id": collection_id},
        )


class IndexingError(MCPError):
    """Raised when document indexing fails."""
    
    def __init__(self, source: str, reason: str, stage: Optional[str] = None):
        suggestions = [
            "Check if the source URL/path is accessible",
            "Verify the document format is supported (PDF, HTML, Markdown, DOCX)",
            "Try with force_reindex=True if the document was partially indexed",
        ]
        
        super().__init__(
            message=f"Failed to index document from '{source}': {reason}",
            error_code="INDEXING_FAILED",
            suggestions=suggestions,
            details={"source": source, "reason": reason, "stage": stage},
        )


class SearchError(MCPError):
    """Raised when search fails."""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Search failed for query '{query[:50]}...': {reason}",
            error_code="SEARCH_FAILED",
            suggestions=[
                "Check if any documents have been indexed",
                "Verify the collection exists",
                "Try a simpler query",
            ],
            details={"query": query, "reason": reason},
        )


class CollectionNotEmptyError(MCPError):
    """Raised when trying to delete a non-empty collection without force."""
    
    def __init__(self, collection_id: str, document_count: int):
        super().__init__(
            message=f"Collection '{collection_id}' contains {document_count} documents.",
            error_code="COLLECTION_NOT_EMPTY",
            suggestions=[
                f"Use delete_collection(name='{collection_id}', force=True) to delete anyway",
                "Delete documents individually first using delete_document()",
            ],
            details={"collection_id": collection_id, "document_count": document_count},
        )


class ValidationError(MCPError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, reason: str, valid_values: Optional[List[str]] = None):
        suggestions = [f"Check the value provided for '{field}'"]
        if valid_values:
            suggestions.append(f"Valid values: {', '.join(valid_values)}")
        
        super().__init__(
            message=f"Invalid value for '{field}': {reason}",
            error_code="VALIDATION_ERROR",
            suggestions=suggestions,
            details={"field": field, "reason": reason},
        )


def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format any exception into a consistent error response.
    
    Args:
        error: The exception to format
        
    Returns:
        Dictionary with error details
    """
    if isinstance(error, MCPError):
        return error.to_dict()
    
    # Handle generic exceptions
    return {
        "error": True,
        "error_code": "INTERNAL_ERROR",
        "message": str(error),
        "suggestions": [
            "This is an unexpected error. Please check the logs for details.",
            "Try the operation again or contact support if the issue persists.",
        ],
    }

