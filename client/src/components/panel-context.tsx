"use client";

import * as React from "react";

// Types matching server responses
export interface Document {
  id: string;
  title: string;
  collection_id: string;
  created_at: string;
  metadata: {
    filename?: string;
    document_type?: string;
    size_bytes?: number;
  };
}

export interface Collection {
  id: string;
  description?: string;
  document_count: number;
  chunk_count: number;
  last_updated: string;
}

// Extended PanelView to support action forms
type PanelView =
  | { type: "none" }
  | { type: "document"; document: Document }
  | { type: "collection"; collection: Collection }
  | { type: "search-results"; query: string; results: any[] }
  // Quick action forms
  | { type: "action:upload-file" }
  | { type: "action:create-collection" }
  | { type: "action:index-url" }
  | { type: "action:process-uploads" };

interface DocumentContextType {
  documents: Document[];
  collections: Collection[];
  selectedView: PanelView;
  isLoading: boolean;

  // Selection actions
  selectDocument: (doc: Document) => void;
  selectCollection: (col: Collection) => void;
  showSearchResults: (query: string, results: any[]) => void;
  clearSelection: () => void;

  // Quick action forms
  showUploadForm: () => void;
  showCreateCollectionForm: () => void;
  showIndexUrlForm: () => void;
  showProcessUploadsForm: () => void;

  // Data refresh
  refreshDocuments: () => Promise<void>;
  refreshCollections: () => Promise<void>;
}

const DocumentContext = React.createContext<DocumentContextType | null>(null);

export function useDocuments() {
  const context = React.useContext(DocumentContext);
  if (!context) {
    throw new Error("useDocuments must be used within DocumentProvider");
  }
  return context;
}

export function DocumentProvider({ children }: { children: React.ReactNode }) {
  const [documents, setDocuments] = React.useState<Document[]>([]);
  const [collections, setCollections] = React.useState<Collection[]>([]);
  const [selectedView, setSelectedView] = React.useState<PanelView>({
    type: "none",
  });
  const [isLoading, setIsLoading] = React.useState(false);

  const refreshDocuments = async () => {
    setIsLoading(true);
    // TODO: Replace with actual MCP client calls
    setIsLoading(false);
  };

  const refreshCollections = async () => {
    // TODO: Replace with actual MCP client calls
  };

  const value: DocumentContextType = {
    documents,
    collections,
    selectedView,
    isLoading,
    // Selection
    selectDocument: (doc) =>
      setSelectedView({ type: "document", document: doc }),
    selectCollection: (col) =>
      setSelectedView({ type: "collection", collection: col }),
    showSearchResults: (query, results) =>
      setSelectedView({ type: "search-results", query, results }),
    clearSelection: () => setSelectedView({ type: "none" }),
    // Quick actions
    showUploadForm: () => setSelectedView({ type: "action:upload-file" }),
    showCreateCollectionForm: () =>
      setSelectedView({ type: "action:create-collection" }),
    showIndexUrlForm: () => setSelectedView({ type: "action:index-url" }),
    showProcessUploadsForm: () =>
      setSelectedView({ type: "action:process-uploads" }),
    // Data
    refreshDocuments,
    refreshCollections,
  };

  return (
    <DocumentContext.Provider value={value}>
      {children}
    </DocumentContext.Provider>
  );
}
