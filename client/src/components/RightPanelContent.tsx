"use client";

import * as React from "react";
import { usePanelControl } from "./collapsible-panels";
import { useDocuments } from "./panel-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  X,
  Upload,
  FolderPlus,
  Link,
  Play,
  FileText,
  Folder,
} from "lucide-react";

export function RightPanelContent() {
  const { closePanel } = usePanelControl();
  const { selectedView, clearSelection } = useDocuments();

  const handleClose = () => {
    clearSelection();
    closePanel();
  };

  // Render based on view type
  const renderContent = () => {
    switch (selectedView.type) {
      case "action:upload-file":
        return <UploadFileForm onClose={handleClose} />;
      case "action:create-collection":
        return <CreateCollectionForm onClose={handleClose} />;
      case "action:index-url":
        return <IndexUrlForm onClose={handleClose} />;
      case "action:process-uploads":
        return <ProcessUploadsForm onClose={handleClose} />;
      case "document":
        return (
          <DocumentDetails
            document={selectedView.document}
            onClose={handleClose}
          />
        );
      case "collection":
        return (
          <CollectionDetails
            collection={selectedView.collection}
            onClose={handleClose}
          />
        );
      case "search-results":
        return (
          <SearchResults
            query={selectedView.query}
            results={selectedView.results}
            onClose={handleClose}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="h-full flex flex-col bg-card/50 border-l border-border/50">
      {renderContent()}
    </div>
  );
}

// ============================================
// Upload File Form (matching your image)
// ============================================
function UploadFileForm({ onClose }: { onClose: () => void }) {
  const [dragActive, setDragActive] = React.useState(false);
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [formData, setFormData] = React.useState({
    title: "",
    collection: "",
    description: "",
  });
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Call index_document with file
    console.log("Uploading:", { file: selectedFile, ...formData });
  };

  return (
    <>
      {/* Header */}
      <div className="flex items-start justify-between p-6 border-b border-border/50">
        <div>
          <h2 className="text-xl font-semibold">Upload Document</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Upload a file to index into your documentation.
            <br />
            Supported formats: PDF, Markdown, DOCX, HTML
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 shrink-0"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="flex-1 overflow-auto p-6 space-y-6"
      >
        {/* File Upload Zone */}
        <div className="space-y-2">
          <label className="text-sm font-medium">File</label>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`
              relative flex flex-col items-center justify-center gap-3 p-8
              border-2 border-dashed rounded-xl cursor-pointer transition-all
              ${
                dragActive
                  ? "border-primary bg-primary/5"
                  : selectedFile
                  ? "border-primary/50 bg-primary/5"
                  : "border-border/50 bg-card/30 hover:border-primary/30 hover:bg-card/50"
              }
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.md,.docx,.html,.txt"
              onChange={handleFileSelect}
              className="hidden"
            />

            <div className="p-3 bg-background rounded-lg border border-border/50">
              {selectedFile ? (
                <FileText className="h-8 w-8 text-primary" />
              ) : (
                <Folder className="h-8 w-8 text-muted-foreground" />
              )}
            </div>

            {selectedFile ? (
              <div className="text-center">
                <p className="font-medium text-sm">{selectedFile.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
            ) : (
              <>
                <p className="font-medium text-sm">
                  Drag and drop file to upload
                </p>
                <p className="text-xs text-muted-foreground">
                  Maximum size: 50 MB
                </p>
              </>
            )}

            <Button
              type="button"
              variant="secondary"
              size="sm"
              className="mt-2"
            >
              Select File
            </Button>
          </div>
        </div>

        {/* Document Title */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Document Title</label>
          <Input
            placeholder="The title displayed in search results"
            value={formData.title}
            onChange={(e) =>
              setFormData({ ...formData, title: e.target.value })
            }
            className="bg-card/50 border-border/50"
          />
        </div>

        {/* Collection */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Collection</label>
          <Input
            placeholder="Collection ID to index into (e.g., design-docs)"
            value={formData.collection}
            onChange={(e) =>
              setFormData({ ...formData, collection: e.target.value })
            }
            className="bg-card/50 border-border/50"
          />
        </div>

        {/* Description */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Description</label>
          <Input
            placeholder="Optional description or notes"
            value={formData.description}
            onChange={(e) =>
              setFormData({ ...formData, description: e.target.value })
            }
            className="bg-card/50 border-border/50"
          />
        </div>

        {/* Submit */}
        <div className="pt-4">
          <Button
            type="submit"
            className="w-full"
            disabled={!selectedFile || !formData.collection}
          >
            <Upload className="h-4 w-4 mr-2" />
            Index Document
          </Button>
        </div>
      </form>
    </>
  );
}

// ============================================
// Create Collection Form
// ============================================
function CreateCollectionForm({ onClose }: { onClose: () => void }) {
  const [formData, setFormData] = React.useState({
    name: "",
    description: "",
    chunkSize: "512",
    chunkOverlap: "50",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Call create_collection
    console.log("Creating collection:", formData);
  };

  return (
    <>
      <div className="flex items-start justify-between p-6 border-b border-border/50">
        <div>
          <h2 className="text-xl font-semibold">New Collection</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Create a new collection to organize your documents.
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <form
        onSubmit={handleSubmit}
        className="flex-1 overflow-auto p-6 space-y-6"
      >
        {/* Icon/Preview */}
        <div className="flex justify-center">
          <div className="p-6 bg-card/50 rounded-2xl border-2 border-dashed border-border/50">
            <FolderPlus className="h-12 w-12 text-primary" />
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Collection Name</label>
          <Input
            placeholder="e.g., design-docs, api-specs"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            className="bg-card/50 border-border/50"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Description</label>
          <Input
            placeholder="Describe what this collection contains"
            value={formData.description}
            onChange={(e) =>
              setFormData({ ...formData, description: e.target.value })
            }
            className="bg-card/50 border-border/50"
          />
        </div>

        {/* Advanced Settings */}
        <details className="group">
          <summary className="text-sm font-medium cursor-pointer text-muted-foreground hover:text-foreground">
            Advanced Settings
          </summary>
          <div className="mt-4 space-y-4 pl-4 border-l-2 border-border/30">
            <div className="space-y-2">
              <label className="text-sm text-muted-foreground">
                Chunk Size (tokens)
              </label>
              <Input
                type="number"
                placeholder="512"
                value={formData.chunkSize}
                onChange={(e) =>
                  setFormData({ ...formData, chunkSize: e.target.value })
                }
                className="bg-card/50 border-border/50"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-muted-foreground">
                Chunk Overlap (tokens)
              </label>
              <Input
                type="number"
                placeholder="50"
                value={formData.chunkOverlap}
                onChange={(e) =>
                  setFormData({ ...formData, chunkOverlap: e.target.value })
                }
                className="bg-card/50 border-border/50"
              />
            </div>
          </div>
        </details>

        <div className="pt-4">
          <Button type="submit" className="w-full" disabled={!formData.name}>
            <FolderPlus className="h-4 w-4 mr-2" />
            Create Collection
          </Button>
        </div>
      </form>
    </>
  );
}

// ============================================
// Index URL Form
// ============================================
function IndexUrlForm({ onClose }: { onClose: () => void }) {
  const [formData, setFormData] = React.useState({
    url: "",
    collection: "",
    title: "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Call index_document with URL
    console.log("Indexing URL:", formData);
  };

  return (
    <>
      <div className="flex items-start justify-between p-6 border-b border-border/50">
        <div>
          <h2 className="text-xl font-semibold">Index from URL</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Index a document directly from a web URL.
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <form
        onSubmit={handleSubmit}
        className="flex-1 overflow-auto p-6 space-y-6"
      >
        <div className="flex justify-center">
          <div className="p-6 bg-card/50 rounded-2xl border-2 border-dashed border-border/50">
            <Link className="h-12 w-12 text-primary" />
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">URL</label>
          <Input
            type="url"
            placeholder="https://example.com/document.pdf"
            value={formData.url}
            onChange={(e) => setFormData({ ...formData, url: e.target.value })}
            className="bg-card/50 border-border/50"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Collection</label>
          <Input
            placeholder="Collection ID to index into"
            value={formData.collection}
            onChange={(e) =>
              setFormData({ ...formData, collection: e.target.value })
            }
            className="bg-card/50 border-border/50"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Title (optional)</label>
          <Input
            placeholder="Override document title"
            value={formData.title}
            onChange={(e) =>
              setFormData({ ...formData, title: e.target.value })
            }
            className="bg-card/50 border-border/50"
          />
        </div>

        <div className="pt-4">
          <Button
            type="submit"
            className="w-full"
            disabled={!formData.url || !formData.collection}
          >
            <Link className="h-4 w-4 mr-2" />
            Index URL
          </Button>
        </div>
      </form>
    </>
  );
}

// ============================================
// Process Uploads Form
// ============================================
function ProcessUploadsForm({ onClose }: { onClose: () => void }) {
  const [isProcessing, setIsProcessing] = React.useState(false);
  const [uploadFiles, setUploadFiles] = React.useState<string[]>([]);

  const handleProcess = async () => {
    setIsProcessing(true);
    // TODO: Call process_uploads
    setTimeout(() => setIsProcessing(false), 2000);
  };

  return (
    <>
      <div className="flex items-start justify-between p-6 border-b border-border/50">
        <div>
          <h2 className="text-xl font-semibold">Process Uploads</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Process all files in the uploads folder.
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-6 space-y-6">
        <div className="flex justify-center">
          <div className="p-6 bg-card/50 rounded-2xl border-2 border-dashed border-border/50">
            <Play className="h-12 w-12 text-primary" />
          </div>
        </div>

        <div className="bg-card/30 rounded-lg p-4 border border-border/50">
          <p className="text-sm text-muted-foreground">
            Files waiting in uploads folder:{" "}
            <span className="text-foreground font-medium">
              {uploadFiles.length || "None"}
            </span>
          </p>
        </div>

        <Button
          onClick={handleProcess}
          className="w-full"
          disabled={isProcessing}
        >
          {isProcessing ? (
            <>Processing...</>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Process All Files
            </>
          )}
        </Button>
      </div>
    </>
  );
}

// Placeholder components for other views
function DocumentDetails({
  document,
  onClose,
}: {
  document: any;
  onClose: () => void;
}) {
  return (
    <div className="p-6">
      <div className="flex justify-between">
        <h2 className="text-xl font-semibold">Document Details</h2>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <p className="text-muted-foreground mt-4">Document: {document.title}</p>
    </div>
  );
}

function CollectionDetails({
  collection,
  onClose,
}: {
  collection: any;
  onClose: () => void;
}) {
  return (
    <div className="p-6">
      <div className="flex justify-between">
        <h2 className="text-xl font-semibold">Collection Details</h2>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <p className="text-muted-foreground mt-4">Collection: {collection.id}</p>
    </div>
  );
}

function SearchResults({
  query,
  results,
  onClose,
}: {
  query: string;
  results: any[];
  onClose: () => void;
}) {
  return (
    <div className="p-6">
      <div className="flex justify-between">
        <h2 className="text-xl font-semibold">Search Results</h2>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <p className="text-muted-foreground mt-4">Query: "{query}"</p>
    </div>
  );
}
