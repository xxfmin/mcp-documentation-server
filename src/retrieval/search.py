from typing import List, Optional, Dict, Any, Tuple
import logging

from lib import DocumentChunk, SearchResult
from storage import VectorStore, DocumentStore

logger = logging.getLogger(__name__)

"""Semantic search engine for document chunks with context expansion and filtering"""
class SearchEngine:

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        document_store: Optional[DocumentStore] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.document_store = document_store or DocumentStore()

        logger.info("SearchEngine initialized")

    def search(
        self,
        query: str,
        collection_ids: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_context: bool = False,
    ) -> List[SearchResult]:
        """
        Search documents using vector similarity

        Args:
            query: Natural language search query
            collection_ids: Optional list of collection IDs to search 
                          (searches all if None)
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"document_type": "pdf"})
            include_context: If True, include adjacent chunks as context

        Returns:
            List of SearchResult objects with chunks, scores, and optional context
        """
        try:
            all_results = []

            # If collection_ids specified, search each collection
            if collection_ids:
                for collection_id in collection_ids:
                    # Build filters for this collection
                    collection_filters = filters.copy() if filters else {}
                    collection_filters["collection_id"] = collection_id

                    # Search this collection and get chunks with scores
                    chunks_with_scores = self.search_with_scores(
                        query=query,
                        collection_id=collection_id,
                        top_k=top_k * 2,  # Get more results per collection for merging
                        filters=collection_filters,
                    )

                    # Convert to SearchResult with context if requested
                    for chunk, score in chunks_with_scores:
                        context_before = None
                        context_after = None
                        if include_context:
                            context_before, context_after = self.get_adjacent_chunks(
                                chunk
                            )

                        result = SearchResult(
                            chunk=chunk,
                            score=score,
                            context_before=context_before,
                            context_after=context_after,
                        )
                        all_results.append(result)
            else:
                # Search all collections - need to find all existing collections
                # Try to list all tables in the database and search each one
                try:
                    # Get all table names from the database
                    table_names = self.vector_store.db.table_names()
                    collection_tables = [name for name in table_names if name.startswith("chunks_")]
                    
                    if not collection_tables:
                        logger.warning("No collection tables found in database")
                        return []
                    
                    logger.info(f"Searching across {len(collection_tables)} collections: {collection_tables}")
                    
                    # Search each collection
                    for table_name in collection_tables:
                        # Extract collection_id from table name (chunks_<collection_id>)
                        collection_id = table_name.replace("chunks_", "") if table_name.startswith("chunks_") else None
                        
                        if collection_id:
                            try:
                                chunks_with_scores = self.search_with_scores(
                                    query=query,
                                    collection_id=collection_id,
                                    top_k=top_k * 2,  # Get more results per collection for merging
                                    filters=filters,
                                )
                                
                                for chunk, score in chunks_with_scores:
                                    context_before = None
                                    context_after = None
                                    if include_context:
                                        context_before, context_after = self.get_adjacent_chunks(chunk)
                                    
                                    result = SearchResult(
                                        chunk=chunk,
                                        score=score,
                                        context_before=context_before,
                                        context_after=context_after,
                                    )
                                    all_results.append(result)
                            except Exception as e:
                                logger.warning(f"Failed to search collection {collection_id}: {e}")
                                continue
                except Exception as e:
                    logger.error(f"Failed to list collections: {e}")
                    # Fallback: try searching default collection
                    chunks_with_scores = self.search_with_scores(
                        query=query,
                        collection_id=None,
                        top_k=top_k,
                        filters=filters,
                    )
                    
                    for chunk, score in chunks_with_scores:
                        context_before = None
                        context_after = None
                        if include_context:
                            context_before, context_after = self.get_adjacent_chunks(chunk)
                        
                        result = SearchResult(
                            chunk=chunk,
                            score=score,
                            context_before=context_before,
                            context_after=context_after,
                        )
                        all_results.append(result)

            # Sort by score (highest first) and limit to top_k
            all_results.sort(key=lambda r: r.score, reverse=True)
            results = all_results[:top_k]

            logger.info(
                f"Search returned {len(results)} results for query: {query[:50]}"
            )
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    def search_with_scores(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search and return chunks with similarity scores

        This method extracts scores from LanceDB results, maintaining separation
        of concerns (VectorStore doesn't need to know about scores).

        Args:
            query: Search query
            collection_id: Optional collection ID
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            List of (DocumentChunk, score) tuples
        """
        try:
            table = self.vector_store.get_table(collection_id)
            
            # Embed the query text before searching
            if not query or not query.strip():
                # For empty queries, use a zero vector
                dim = self.vector_store.embedder.get_embedding_dimension()
                query_embedding = [0.0] * dim
            else:
                # Embed the query text
                query_embedding = self.vector_store.embedder.embed_text(query)
            
            # Build search query - simple approach like the example
            search_query = table.search(query_embedding).limit(top_k)
            
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
                
                if where_clauses:
                    where_expr = " AND ".join(where_clauses)
                    search_query = search_query.where(where_expr)
            
            # Execute search - simple approach like the example
            # Just get the results directly from LanceDB
            results = search_query.to_pandas()
            
            # Extract chunks and use raw distance as score (lower distance = better match)
            # Simple conversion: just use 1 / (1 + distance) for a basic similarity score
            chunks_with_scores = []
            for _, row in results.iterrows():
                chunk = self.vector_store.row_to_chunk(row)
                if chunk:
                    distance = row.get('_distance', 1.0)
                    # Simple score: lower distance = higher score
                    score = 1.0 / (1.0 + distance)
                    chunks_with_scores.append((chunk, score))
            
            return chunks_with_scores
            
        except Exception as e:
            # Handle case where table doesn't exist (no documents indexed yet)
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                table_name = self.vector_store.get_table_name(collection_id)
                logger.info(f"Table {table_name} does not exist yet. No documents have been indexed.")
                return []
            logger.error(f"Search with scores failed: {e}")
            raise

    def get_adjacent_chunks(
        self, chunk: DocumentChunk
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get adjacent chunks for context expansion

        Args:
            chunk: Current chunk

        Returns:
            Tuple of (context_before, context_after) text
        """
        try:
            document_id = chunk.metadata.document_id
            collection_id = chunk.metadata.collection_id
            chunk_index = chunk.metadata.chunk_index

            # Get all chunks for this document (more efficient than searching)
            # Use filters to get all chunks, then sort by index
            chunks_with_scores = self.search_with_scores(
                query="",  # Empty query - we'll filter by document_id
                collection_id=collection_id,
                top_k=10000,  # Large number to get all chunks
                filters={"document_id": document_id},
            )

            # Extract just the chunks and sort by chunk_index
            all_chunks = [c for c, _ in chunks_with_scores]
            all_chunks.sort(key=lambda c: c.metadata.chunk_index)

            # Find current chunk position
            current_pos = None
            for i, c in enumerate(all_chunks):
                if c.chunk_id == chunk.chunk_id:
                    current_pos = i
                    break

            if current_pos is None:
                return None, None

            # Get previous chunk
            context_before = None
            if current_pos > 0:
                prev_chunk = all_chunks[current_pos - 1]
                context_before = prev_chunk.text

            # Get next chunk
            context_after = None
            if current_pos < len(all_chunks) - 1:
                next_chunk = all_chunks[current_pos + 1]
                context_after = next_chunk.text

            return context_before, context_after

        except Exception as e:
            logger.debug(f"Failed to get adjacent chunks: {e}")
            return None, None

    def search_by_document(
        self,
        document_id: str,
        query: Optional[str] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search within a specific document.

        Args:
            document_id: Document ID to search within
            query: Optional search query (returns all chunks if None)
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        try:
            # Get document to find collection_id
            doc = self.document_store.get_document(document_id)
            if not doc:
                logger.warning(f"Document {document_id} not found")
                return []

            collection_id = doc.collection_id

            # Build filters
            filters = {"document_id": document_id}

            # Search with scores
            if query:
                chunks_with_scores = self.search_with_scores(
                    query=query,
                    collection_id=collection_id,
                    top_k=top_k,
                    filters=filters,
                )
            else:
                # Get all chunks for document (no query, sorted by index)
                chunks_with_scores = self.search_with_scores(
                    query="",  # Empty query
                    collection_id=collection_id,
                    top_k=top_k,
                    filters=filters,
                )
                # Sort by chunk_index instead of score when no query
                chunks_with_scores.sort(
                    key=lambda x: x[0].metadata.chunk_index
                )

            # Convert to SearchResult
            results = []
            for chunk, score in chunks_with_scores:
                result = SearchResult(chunk=chunk, score=score)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to search document {document_id}: {e}", exc_info=True)
            return []

