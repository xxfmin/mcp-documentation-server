"use client";

import * as React from "react";
import { mcpClient } from "@/lib/mcp-client";
import { Document, Collection, SearchResult } from "@/lib/mcp-types";

// Re-export types for other components
export type { Document, Collection, SearchResult };

// Extended PanelView to support action forms
type PanelView =
  | { type: "none" }
  | { type: "document"; document: Document }
  | { type: "collection"; collection: Collection }
  | { type: "search-results"; query: string; results: SearchResult[] }
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

  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;

  // Selection actions
  selectDocument: (doc: Document) => void;
  selectCollection: (col: Collection) => void;
  showSearchResults: (query: string, results: SearchResult[]) => void;
  clearSelection: () => void;

  // Quick action forms
  showUploadForm: () => void;
  showCreateCollectionForm: () => void;
  showIndexUrlForm: () => void;
  showProcessUploadsForm: () => void;

  // Data refresh
  refreshDocuments: () => Promise<void>;
  refreshCollections: () => Promise<void>;

  // Connection control
  connect: () => Promise<void>;
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

  // Connection and error state
  const [isConnected, setIsConnected] = React.useState(false);
  const [isConnecting, setIsConnecting] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  // Use ref to track connection state without triggering re-renders
  const connectionRef = React.useRef({
    isConnecting: false,
    isConnected: false,
  });

  // Connect to MCP server
  const connect = React.useCallback(async () => {
    if (
      connectionRef.current.isConnected ||
      connectionRef.current.isConnecting
    ) {
      return;
    }

    connectionRef.current.isConnecting = true;
    setIsConnecting(true);
    setError(null);

    try {
      await mcpClient.connect();
      connectionRef.current.isConnected = true;
      connectionRef.current.isConnecting = false;
      setIsConnected(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to connect";
      connectionRef.current.isConnecting = false;
      setError(message);
      setIsConnected(false);
    } finally {
      setIsConnecting(false);
    }
  }, []);

  // Fetch documents from server
  const refreshDocuments = React.useCallback(async () => {
    if (!isConnected) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await mcpClient.listDocuments();
      if (response?.documents) {
        setDocuments(response.documents);
      } else {
        setDocuments([]);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to fetch documents";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [isConnected]);

  // Fetch collections from server
  const refreshCollections = React.useCallback(async () => {
    if (!isConnected) {
      return;
    }

    try {
      const response = await mcpClient.listCollections();
      if (response?.collections) {
        setCollections(response.collections);
      } else {
        setCollections([]);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to fetch collections";
      setError(message);
    }
  }, [isConnected]);

  // Connect on mount (client-side only)
  React.useEffect(() => {
    if (typeof window !== "undefined") {
      connect();
    }

    return () => {
      connectionRef.current.isConnecting = false;
      connectionRef.current.isConnected = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch data when connected
  React.useEffect(() => {
    if (isConnected) {
      Promise.all([refreshDocuments(), refreshCollections()]);
    }
  }, [isConnected, refreshDocuments, refreshCollections]);

  const value: DocumentContextType = {
    documents,
    collections,
    selectedView,
    isLoading,
    // Connection state
    isConnected,
    isConnecting,
    error,
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
    connect,
  };

  return (
    <DocumentContext.Provider value={value}>
      {children}
    </DocumentContext.Provider>
  );
}
