import {
  Collection,
  IndexDocumentResponse,
  JSONRPCRequest,
  JSONRPCResponse,
  ListCollectionsResponse,
  ListDocumentsResponse,
  MCPErrorResponse,
  SearchResponse,
  UploadFile,
} from "./mcp-types";

type ConnectionState = "disconnected" | "connecting" | "connected" | "error";

export class MCPClient {
  private baseUrl: string;
  private eventSource: EventSource | null = null;
  private messageEndpoint: string | null = null;
  private pendingRequests: Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
    }
  > = new Map();
  private requestId = 0;
  private connectionState: ConnectionState = "disconnected";
  private onStateChange?: (state: ConnectionState) => void;
  private pendingConnectionCleanup: (() => void) | null = null;

  constructor(baseUrl: string = "http://127.0.0.1:8000") {
    this.baseUrl = baseUrl;
  }

  setStateChangeHandler(handler: (state: ConnectionState) => void) {
    this.onStateChange = handler;
  }

  private updateState(state: ConnectionState) {
    this.connectionState = state;
    this.onStateChange?.(state);
  }

  async connect(): Promise<void> {
    if (this.eventSource) {
      return; // Already connected
    }

    this.updateState("connecting");

    return new Promise((resolve, reject) => {
      let connectionTimeout: NodeJS.Timeout;
      let resolved = false;

      const cleanup = () => {
        if (connectionTimeout) {
          clearTimeout(connectionTimeout);
        }
        this.pendingConnectionCleanup = null;
      };

      this.pendingConnectionCleanup = () => {
        if (!resolved) {
          resolved = true;
          cleanup();
        }
      };

      // Set a timeout for connection (10 seconds)
      connectionTimeout = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          cleanup();
          const error = new Error(
            `Connection timeout: Failed to connect to ${this.baseUrl}/sse after 10 seconds.`
          );
          this.updateState("error");
          if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
          }
          reject(error);
        }
      }, 10000);

      // Create EventSource for SSE connection
      const sseUrl = `${this.baseUrl}/sse`;

      try {
        this.eventSource = new EventSource(sseUrl);
      } catch (err) {
        cleanup();
        const error = new Error(`Failed to create EventSource: ${err}`);
        this.updateState("error");
        reject(error);
        return;
      }

      // Handle the "endpoint" event - server tells us where to POST messages
      this.eventSource.addEventListener("endpoint", (event) => {
        if (resolved) return;
        resolved = true;
        cleanup();

        // The data is a relative URL like "/messages/?session_id=abc123"
        this.messageEndpoint = `${this.baseUrl}${event.data}`;

        // Perform MCP initialization handshake
        this.performInitialization()
          .then(() => {
            this.updateState("connected");
            resolve();
          })
          .catch((err) => {
            this.updateState("error");
            reject(err);
          });
      });

      // Handle "message" events - server responses to our requests
      this.eventSource.addEventListener("message", (event) => {
        try {
          const response: JSONRPCResponse = JSON.parse(event.data);
          this.handleResponse(response);
        } catch (error) {
          console.error("[MCP] Failed to parse message:", error);
        }
      });

      // Handle errors
      this.eventSource.onerror = () => {
        if (this.eventSource?.readyState === EventSource.CLOSED) {
          if (!resolved) {
            resolved = true;
            cleanup();
            const err = new Error(
              `Connection closed: Unable to connect to ${this.baseUrl}/sse. ` +
                `Server may be down or CORS is blocking the connection.`
            );
            this.updateState("error");
            reject(err);
          }
        }
      };
    });
  }

  disconnect() {
    if (this.pendingConnectionCleanup) {
      this.pendingConnectionCleanup();
      this.pendingConnectionCleanup = null;
    }

    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.messageEndpoint = null;
      this.updateState("disconnected");
    }
  }

  private handleResponse(response: JSONRPCResponse) {
    const pending = this.pendingRequests.get(response.id as number);
    if (!pending) {
      return;
    }

    this.pendingRequests.delete(response.id as number);

    if (response.error) {
      pending.reject(
        new Error(
          `MCP Error: ${response.error.message} (${response.error.code})`
        )
      );
    } else {
      pending.resolve(response.result);
    }
  }

  /**
   * Perform MCP initialization handshake.
   */
  private async performInitialization(): Promise<void> {
    // Step 1: Send initialize request
    await this.sendRequest("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {
        tools: {},
      },
      clientInfo: {
        name: "mcp-doc-client",
        version: "1.0.0",
      },
    });

    // Step 2: Send initialized notification (no response expected)
    await this.sendNotification("notifications/initialized", {});
  }

  /**
   * Send a JSON-RPC request and wait for response.
   */
  private async sendRequest<T>(
    method: string,
    params: Record<string, unknown>
  ): Promise<T> {
    if (!this.messageEndpoint) {
      throw new Error("Not connected to MCP server");
    }

    const id = ++this.requestId;
    const request: JSONRPCRequest = {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };

    const responsePromise = new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
    });

    const response = await fetch(this.messageEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      this.pendingRequests.delete(id);
      throw new Error(`HTTP error: ${response.status}`);
    }

    return responsePromise;
  }

  /**
   * Send a JSON-RPC notification (no response expected).
   */
  private async sendNotification(
    method: string,
    params: Record<string, unknown>
  ): Promise<void> {
    if (!this.messageEndpoint) {
      throw new Error("Not connected to MCP server");
    }

    const notification = {
      jsonrpc: "2.0",
      method,
      params,
    };

    const response = await fetch(this.messageEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(notification),
    });

    if (!response.ok) {
      throw new Error(`HTTP error sending notification: ${response.status}`);
    }
  }

  async callTool<T>(
    name: string,
    args: Record<string, unknown> = {}
  ): Promise<T> {
    if (!this.messageEndpoint) {
      throw new Error("Not connected to MCP server");
    }

    const id = ++this.requestId;

    const request: JSONRPCRequest = {
      jsonrpc: "2.0",
      id,
      method: "tools/call",
      params: {
        name,
        arguments: args,
      },
    };

    const responsePromise = new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(id, {
        resolve: (rawResult: unknown) => {
          // MCP tool responses have format: { content: [{ type: "text", text: "..." }] }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const mcpResult = rawResult as any;

          if (mcpResult?.content && Array.isArray(mcpResult.content)) {
            const textContent = mcpResult.content.find(
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (item: any) => item.type === "text"
            );

            if (textContent?.text) {
              try {
                const parsed = JSON.parse(textContent.text);
                resolve(parsed as T);
                return;
              } catch {
                // Fall through to raw result
              }
            }
          }

          resolve(rawResult as T);
        },
        reject,
      });
    });

    try {
      const response = await fetch(this.messageEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
    } catch (error) {
      this.pendingRequests.delete(id);
      throw error;
    }

    return responsePromise;
  }

  getState(): ConnectionState {
    return this.connectionState;
  }

  isConnected(): boolean {
    return this.connectionState === "connected";
  }

  // Collection Management
  async listCollections(): Promise<ListCollectionsResponse> {
    return this.callTool<ListCollectionsResponse>("list_collections");
  }

  async getCollection(name: string): Promise<Collection | MCPErrorResponse> {
    return this.callTool<Collection | MCPErrorResponse>("get_collection", {
      name,
    });
  }

  async createCollection(params: {
    name: string;
    description?: string;
    chunk_size?: number;
    chunk_overlap?: number;
  }): Promise<{ collection_id: string; status: string } | MCPErrorResponse> {
    return this.callTool("create_collection", params);
  }

  async deleteCollection(
    name: string,
    force: boolean = false
  ): Promise<{ collection_id: string; status: string } | MCPErrorResponse> {
    return this.callTool("delete_collection", { name, force });
  }

  // Document Management
  async listDocuments(collectionId?: string): Promise<ListDocumentsResponse> {
    return this.callTool<ListDocumentsResponse>("list_documents", {
      collection_id: collectionId,
    });
  }

  async getDocument(documentId: string): Promise<Document | MCPErrorResponse> {
    return this.callTool<Document | MCPErrorResponse>("get_document", {
      document_id: documentId,
    });
  }

  async indexDocument(params: {
    collection_id: string;
    source: string;
    document_type?: string;
    metadata?: Record<string, unknown>;
    force_reindex?: boolean;
  }): Promise<IndexDocumentResponse> {
    return this.callTool<IndexDocumentResponse>("index_document", params);
  }

  async deleteDocument(
    documentId: string
  ): Promise<{ document_id: string; status: string } | MCPErrorResponse> {
    return this.callTool("delete_document", { document_id: documentId });
  }

  // Search
  async searchDocuments(params: {
    query: string;
    document_id?: string;
    collection_ids?: string[];
    top_k?: number;
    include_context?: boolean;
  }): Promise<SearchResponse> {
    return this.callTool<SearchResponse>("search_documents", params);
  }

  async hybridSearch(params: {
    query: string;
    document_id?: string;
    collection_ids?: string[];
    top_k?: number;
    include_context?: boolean;
    vector_weight?: number;
    keyword_weight?: number;
    rerank?: boolean;
    rerank_method?: string;
  }): Promise<SearchResponse> {
    return this.callTool<SearchResponse>("hybrid_search", params);
  }

  // File Upload
  async listUploadsFiles(): Promise<{ files: UploadFile[] }> {
    return this.callTool<{ files: UploadFile[] }>("list_uploads_files");
  }

  async processUploads(): Promise<{ processed: number; errors: string[] }> {
    return this.callTool("process_uploads");
  }

  async getUploadsPath(): Promise<string> {
    return this.callTool<string>("get_uploads_path");
  }

  // System
  async getServerInfo(): Promise<{
    name: string;
    version: string;
    capabilities: string[];
    tools: string[];
  }> {
    return this.callTool("get_server_info");
  }

  async getSystemStats(): Promise<Record<string, unknown>> {
    return this.callTool("get_system_stats");
  }
}

export const mcpClient = new MCPClient(
  process.env.NEXT_PUBLIC_MCP_SERVER_URL || "http://127.0.0.1:8000"
);
