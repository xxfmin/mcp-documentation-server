from typing import List
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc.document import DoclingDocument
import tiktoken
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib import (
    Config,
    DocumentChunk,
    ChunkMetadata,
    DocumentMetadata,
    generate_chunk_id,
)

logger = logging.getLogger(__name__)

"""Handles document chunking with rich metadata"""
class Chunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap:int = 50,
        merge_peers: bool = True
    ):
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4o"),
            max_tokens=chunk_size
        )

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=merge_peers
        )

        self.chunk_overlap = chunk_overlap
        self.use_parallel = Config.PARALLEL_ENABLED
        self.max_workers = Config.MAX_WORKERS
    
    def chunk_document(
        self,
        doc: DoclingDocument,
        doc_metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        try:
            # Apply chunking
            chunk_iter = self.chunker.chunk(dl_doc=doc)
            raw_chunks = list(chunk_iter)

            if not raw_chunks:
                logger.warning(f"No chunks generated for document {doc_metadata.document_id}")
                return []
            
            logger.info(f"Generated {len(raw_chunks)} chunks")

            # Use parallel processing for large documents if enabled
            if self.use_parallel and len(raw_chunks) > 100:
                return self.chunk_parallel(raw_chunks, doc_metadata)
            else:
                return self.chunk_sequential(raw_chunks, doc_metadata)
        except Exception as e:
            logger.error(f"Failed to chunk document: {e}", exc_info = True)
            raise

    def chunk_sequential(
            self,
            raw_chunks: List,
            doc_metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        document_chunks = []
        for idx, raw_chunk in enumerate(raw_chunks):
            chunk = self.create_chunk(raw_chunk, idx, len(raw_chunks), doc_metadata)
            document_chunks.append(chunk)
        return document_chunks
    
    def chunk_parallel(
            self,
            raw_chunks: List,
            doc_metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        logger.info(f"Using parallel chunking with {self.max_workers} workers")
        document_chunks: List[DocumentChunk | None] = [None] * len(raw_chunks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.create_chunk,
                    raw_chunk,
                    idx,
                    len(raw_chunks),
                    doc_metadata
                ): idx
                for idx, raw_chunk in enumerate(raw_chunks)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    document_chunks[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing chunk {idx}: {e}")

        return [chunk for chunk in document_chunks if chunk is not None]
    
    def create_chunk(
        self,
        raw_chunk,
        idx: int,
        total: int,
        doc_metadata: DocumentMetadata
    ) -> DocumentChunk:
        chunk_id = generate_chunk_id(doc_metadata.document_id, idx)

        # Extract structural metadata
        section_hierarchy = self.extract_section_hierarchy(raw_chunk)
        page_numbers = self.extract_page_numbers(raw_chunk)
        char_start, char_end = self.extract_char_span(raw_chunk)

        # Create chunk metadata
        chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_metadata.document_id,
            collection_id=doc_metadata.collection_id,
            chunk_index=idx,
            total_chunks=total,
            char_start=char_start,
            char_end=char_end,
            page_numbers=page_numbers,
            section_hierarchy=section_hierarchy,
            heading_level=len(section_hierarchy) if section_hierarchy else None,
            filename=doc_metadata.filename,
            document_type=doc_metadata.document_type,
            source_url=doc_metadata.source_url,
            indexed_at=datetime.now(timezone.utc),
        )

        return DocumentChunk(
            chunk_id=chunk_id,
            text=raw_chunk.text,
            metadata=chunk_metadata,
        )
    
    def extract_section_hierarchy(self, raw_chunk) -> List[str]:
        """Extract heading hierarchy from chunk."""
        try:
            if hasattr(raw_chunk, "meta") and raw_chunk.meta:
                if hasattr(raw_chunk.meta, "headings") and raw_chunk.meta.headings:
                    return list(raw_chunk.meta.headings)
        except (AttributeError, TypeError):
            pass
        return []

    def extract_page_numbers(self, raw_chunk) -> List[int]:
        """Extract page numbers from chunk."""
        page_numbers = []
        try:
            if hasattr(raw_chunk, "meta") and raw_chunk.meta:
                if hasattr(raw_chunk.meta, "doc_items") and raw_chunk.meta.doc_items:
                    for item in raw_chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no") and prov.page_no is not None:
                                    page_numbers.append(prov.page_no)
        except (AttributeError, TypeError):
            pass
        return sorted(set(page_numbers)) if page_numbers else []

    def extract_char_span(self, raw_chunk) -> tuple[int, int]:
        """Extract character span from chunk."""
        char_start = 0
        char_end = len(raw_chunk.text) if hasattr(raw_chunk, "text") else 0

        try:
            if hasattr(raw_chunk, "meta") and raw_chunk.meta:
                if hasattr(raw_chunk.meta, "doc_items") and raw_chunk.meta.doc_items:
                    first_item = raw_chunk.meta.doc_items[0]
                    if hasattr(first_item, "prov") and first_item.prov:
                        first_prov = first_item.prov[0]
                        if hasattr(first_prov, "charspan") and first_prov.charspan:
                            char_start = first_prov.charspan[0]

                    last_item = raw_chunk.meta.doc_items[-1]
                    if hasattr(last_item, "prov") and last_item.prov:
                        last_prov = last_item.prov[-1]
                        if hasattr(last_prov, "charspan") and last_prov.charspan:
                            char_end = last_prov.charspan[1]
        except (AttributeError, TypeError, IndexError):
            pass

        return char_start, char_end