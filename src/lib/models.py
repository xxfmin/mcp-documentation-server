from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# Metadata for a source document
class DocumentMetadata(BaseModel):
    document_id: str
    collection_id: str
    filename: str
    document_type: str # pdf, html, markdown, etc
    source_url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    pages: Optional[int] = None
    indexed_at: datetime
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

# Metadata for a document chunk
class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    collection_id: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    page_numbers: Optional[List[int]] = None
    section_hierarchy: List[str] = Field(default_factory=list)
    heading_level: Optional[int] = None
    filename: str
    document_type: str
    source_url: Optional[str] = None
    indexed_at: datetime

# A chunk of text with metadata
class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

# Search result with chunk and score
class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    context_before: Optional[str] = None
    context_after: Optional[str] = None

# Document collection configuration
class Collection(BaseModel):
    collection_id: str
    description: Optional[str] = None
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    chunk_size: int = 512
    chunk_overlap: int = 50
    created_at: datetime
    last_updated: datetime
    document_count: int = 0
    chunk_count: int = 0

# Result of indexing operation
class IndexingResult(BaseModel):
    document_id: str
    collection_id: str
    chunk_count: int
    status: str # indexed, failed, skipped
    error: Optional[str] = None