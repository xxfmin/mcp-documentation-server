"use client";

import * as React from "react";
import {
  FileText,
  FolderPlus,
  Upload,
  Folder,
  FileSpreadsheet,
  Image,
  MoreVertical,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePanelControl } from "./collapsible-panels";
import { useDocuments } from "./panel-context";

const typeIcons: Record<string, React.ElementType> = {
  pdf: FileText,
  docx: FileText,
  xls: FileSpreadsheet,
  xlsx: FileSpreadsheet,
  png: Image,
  jpg: Image,
  jpeg: Image,
  default: FileText,
};

export function LeftPanelContent() {
  const { openPanel } = usePanelControl();
  const {
    documents,
    // If you added these in the context, you can also destructure:
    // isLoading,
    // error,
    showUploadForm,
    showCreateCollectionForm,
    showIndexUrlForm,
    showProcessUploadsForm,
    // Optional: selectDocument if you added it
    selectDocument,
  } = useDocuments();

  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedDocId, setSelectedDocId] = React.useState<string | null>(null);

  // When a row is clicked, select that document in context and open the right panel
  const handleDocumentClick = (docId: string) => {
    const doc = documents.find((d) => d.id === docId);
    if (doc) {
      setSelectedDocId(docId);
      // Tell the context which document is selected (so RightPanel can show details)
      selectDocument?.(doc);
      openPanel();
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // For now, just rely on client-side filtering below.
    // Later, you can wire this up to hybrid_search and show search results.
  };

  const handleUploadClick = () => {
    showUploadForm();
    openPanel();
  };

  const handleCreateCollectionClick = () => {
    showCreateCollectionForm();
    openPanel();
  };

  const handleIndexUrlClick = () => {
    showIndexUrlForm();
    openPanel();
  };

  const handleProcessUploadsClick = () => {
    showProcessUploadsForm();
    openPanel();
  };

  // Derive a simple "type" from document metadata for filtering/icons
  const getDocType = (doc: (typeof documents)[number]) =>
    ((doc.metadata.document_type as string) || "").toLowerCase();

  const filteredDocs = (documents ?? []).filter((doc) => {
    const title = doc.title?.toLowerCase() || "";
    const filename =
      (doc.metadata.filename as string | undefined)?.toLowerCase() || "";
    const q = searchQuery.toLowerCase();

    return !q || title.includes(q) || filename.includes(q);
  });

  return (
    <div className="h-full flex flex-col px-6 py-4 gap-y-8 text-foreground">
      {/* Quick Actions */}
      <div className="flex flex-col gap-y-5">
        <h2 className="font-semibold text-lg">Quick Actions</h2>

        <div className="grid grid-cols-4 gap-3">
          <QuickActionCard
            icon={FileText}
            label="Index URL"
            onClick={handleIndexUrlClick}
          />
          <QuickActionCard
            icon={FolderPlus}
            label="New collection"
            onClick={handleCreateCollectionClick}
          />
          <QuickActionCard
            icon={Upload}
            label="Upload file"
            onClick={handleUploadClick}
          />
          <QuickActionCard
            icon={Folder}
            label="Process uploads"
            onClick={handleProcessUploadsClick}
          />
        </div>
      </div>

      {/* All Files */}
      <div className="flex-1 flex flex-col min-h-0 gap-y-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">All files</h2>

          <form onSubmit={handleSearch} className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-foreground/50" />
            <input
              className="text-xs pl-7 pr-2 py-1 rounded border border-border/50 bg-card/30"
              placeholder="Search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </form>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto rounded-lg border border-border/50">
          <table className="w-full">
            <thead className="bg-card/30 sticky text-foreground/70 top-0">
              <tr className="text-xs">
                <th className="text-left py-3 px-4 font-medium">File name</th>
                <th className="text-left py-3 px-4 font-medium">Collection</th>
                <th className="text-left py-3 px-4 font-medium">Created at</th>
                <th className="w-10"></th>
              </tr>
            </thead>
            <tbody>
              {filteredDocs.length === 0 ? (
                <tr>
                  <td
                    colSpan={4}
                    className="py-6 px-4 text-xs text-foreground/60 text-center"
                  >
                    No documents found. Try indexing a file or URL from the
                    quick actions above.
                  </td>
                </tr>
              ) : (
                filteredDocs.map((doc) => {
                  const docType = getDocType(doc);
                  const Icon = typeIcons[docType] || typeIcons.default;
                  const sizeBytes = doc.metadata.size_bytes as
                    | number
                    | undefined;
                  const sizeLabel =
                    typeof sizeBytes === "number"
                      ? `${(sizeBytes / 1024).toFixed(1)} KB`
                      : undefined;

                  return (
                    <tr
                      key={doc.id}
                      onClick={() => handleDocumentClick(doc.id)}
                      className={
                        "border-t border-border/30 cursor-pointer transition-colors " +
                        (selectedDocId === doc.id
                          ? "bg-foreground/10"
                          : "hover:bg-card/50")
                      }
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-card rounded-md">
                            <Icon className="h-4 w-4 text-foreground" />
                          </div>
                          <div>
                            <div className="font-medium text-sm">
                              {doc.title ||
                                (doc.metadata.filename as string) ||
                                doc.id}
                            </div>
                            {sizeLabel && (
                              <div className="text-xs text-foreground/70">
                                {sizeLabel}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-sm text-foreground">
                          {doc.collection_id}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-xs text-foreground/80">
                        {doc.created_at
                          ? new Date(doc.created_at).toLocaleString()
                          : "—"}
                      </td>
                      <td className="py-3 px-4">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDocumentClick(doc.id);
                          }}
                        >
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function QuickActionCard({
  icon: Icon,
  label,
  onClick,
}: {
  icon: React.ElementType;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex flex-col items-start gap-3 p-4 rounded-xl border border-border/50 bg-card/30 hover:bg-card/60 hover:border-primary/30 transition-all group"
    >
      <div className="p-2 bg-background rounded-lg border border-border/50 group-hover:border-primary/30">
        <Icon className="h-5 w-5 text-foreground" />
      </div>
      <span className="text-sm font-medium">{label}</span>
    </button>
  );
}
