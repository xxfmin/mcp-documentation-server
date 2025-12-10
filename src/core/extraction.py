from typing import Iterator, Optional
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
import logging
from datetime import datetime, timezone
from pathlib import Path

from lib import Config, DocumentMetadata, generate_document_id

logger = logging.getLogger(__name__)

"""Handles document extraction using Docling"""
class DocumentExtractor:
    def __init__(self):
        self.converter = DocumentConverter()
        self.use_streaming = Config.STREAMING_ENABLED

    # Extract a single document
    def extract(
        self,
        source: str,
        collection_id: str,
        custom_metadata: Optional[dict] = None
    ) -> tuple[DoclingDocument, DocumentMetadata]:
        try:
            # Check if streaming is needed
            if self.use_streaming and source.startswith("/"):
                source_path = Path(source)
                if source_path.exists():
                    file_size = source_path.stat().st_size
                    if file_size > Config.STREAM_FILE_SIZE_LIMIT:
                        logger.info(f"Using streaming for large file: {format}")
                        # Streaming extraction would go here
            
            # Extract document 
            result = self.converter.convert(source)
            doc = result.document

            # Generate document ID
            document_id = generate_document_id(source)

            # Extract metadata
            metadata = DocumentMetadata(
                document_id = document_id,
                collection_id = collection_id,
                filename = self.extract_filename(source,doc),
                document_type = self.detect_type(source),
                source_url = source if source.startswith("http") else None, 
                title = self.extract_title(doc),
                pages = self.extract_page_count(doc),
                indexed_at = datetime.now(timezone.utc),
                custom_metadata = custom_metadata or {}
            )

            logger.info(f"Extracted document: {metadata.document_id}")
            return doc, metadata
        except Exception as e:
            logger.error(f"Failed to extract {source}: {e}")
            raise

    # Extract filename from source or document
    def extract_filename(self, source: str, doc: DoclingDocument) -> str:
        if doc.origin is not None:
            filename = getattr(doc.origin, "filename", None)
            if filename:
                return filename
        return source.split("/")[-1]
    
    # Detect document type from source
    def detect_type(self, source: str) -> str:
        source_lower = source.lower()
        if source_lower.endswith(".pdf"):
            return "pdf"
        elif source_lower.endswith((".html", ".htm")) or source.startswith("http"):
            return "html"
        elif source_lower.endswith((".md", ".markdown")):
            return "markdown"
        elif source_lower.endswith(".docx"):
            return "docx"
        return "unknown"
    
    # Extract document title
    def extract_title(self, doc: DoclingDocument) -> Optional[str]:
        if not hasattr(doc, "texts"):
            return None
        texts = getattr(doc, "texts", None)
        if texts:
            for item in texts[:5]:
                text = getattr(item, "text", None)
                if text:
                    return text[:200]
        return None
    
    # Extract page count
    def extract_page_count(self, doc: DoclingDocument) -> Optional[int]:
        pages = getattr(doc, "pages", None)
        if pages is not None:
            return len(pages)
        return None