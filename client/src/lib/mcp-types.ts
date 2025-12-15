
// JSON-RPC 2.0 Types
export interface JSONRPCRequest {
  jsonrpc: "2.0";
  id: number | string;
  method: string;
  params?: Record<string, unknown>;
}

export interface JSONRPCResponse<T = unknown> {
  jsonrpc: "2.0";
  id: number | string;
  result?: T;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

// MCP Tool Response Types
export interface Collection {
  id: string;
  description: string | null;
  document_count: number;
  chunk_count: number;
  last_updated: string;
  embedding_model?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  created_at?: string;
}

export interface Document {
  id: string;
  title: string;
  collection_id: string;
  created_at: string;
  metadata: Record<string, unknown>;
  chunks_count?: number;
}

export interface SearchResult {
  chunk_id: string;
  document_id: string;
  collection_id: string;
  chunk_index: number;
  score: number;
  text: string;
  metadata: {
    filename: string;
    page_numbers: number[] | null;
    section: string | null;
    headings: string[];
  };
  context?: {
    before: string | null;
    after: string | null;
  } | null;
}

export interface UploadFile {
  name: string;
  size: number;
  modified: number;
  supported: boolean;
}

// Tool-specific response types
export interface ListCollectionsResponse {
  collections: Collection[];
}

export interface ListDocumentsResponse {
  documents: Document[];
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
  search_type?: string;
  weights?: { vector: number; keyword: number };
  reranked?: boolean;
}

export interface IndexDocumentResponse {
  document_id: string;
  collection_id: string;
  source: string;
  status: "indexed" | "failed" | "skipped";
  chunk_count?: number;
  pages?: number;
  indexed_at?: string;
  metadata?: Record<string, unknown>;
  error?: string;
}

// Error response
export interface MCPErrorResponse {
  error: string;
  code: string;
  details?: Record<string, unknown>;
  suggestions?: string[];
}