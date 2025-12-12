"use client";

import * as React from "react";
import {
  FileText,
  FolderPlus,
  Upload,
  Folder,
  File,
  FileSpreadsheet,
  Image,
  MoreVertical,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePanelControl } from "./collapsible-panels";
import { useDocuments } from "./panel-context";

const mockDocuments = [
  {
    id: "1",
    title: "Dashboard tech requirements",
    type: "docx",
    size: "220 KB",
    date: "Jan 4, 2024",
    collection: "design-docs",
  },
  {
    id: "2",
    title: "Q4_2025 Reporting",
    type: "pdf",
    size: "1.2 MB",
    date: "Jan 5, 2024",
    collection: "reports",
  },
  {
    id: "3",
    title: "FY_2024-25 Financials",
    type: "xls",
    size: "628 KB",
    date: "Jan 6, 2024",
    collection: "financials",
  },
];

const tabs = ["View all", "Documents", "PDFs", "Images", "Others"] as const;

type TabType = (typeof tabs)[number];

const typeIcons: Record<string, React.ElementType> = {
  pdf: FileText,
  docx: FileText,
  xls: FileSpreadsheet,
  png: Image,
  jpg: Image,
  default: FileText,
};

export function LeftPanelContent() {
  const { openPanel } = usePanelControl();
  const {
    showUploadForm,
    showCreateCollectionForm,
    showIndexUrlForm,
    showProcessUploadsForm,
  } = useDocuments();
  const [activeTab, setActiveTab] = React.useState<TabType>("View all");
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedDocId, setSelectedDocId] = React.useState<string | null>(null);

  const handleDocumentClick = (docId: string) => {
    setSelectedDocId(docId);
    openPanel(); // Opens the right panel with document details
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Call search_documents or hybrid_search
      openPanel(); // Show search results in right panel
    }
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

  const filteredDocs = mockDocuments.filter((doc) => {
    const matchesSearch = doc.title
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesTab =
      activeTab === "View all" || doc.type === activeTab.toLowerCase();
    return matchesSearch && matchesTab;
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
        <h2 className="text-lg font-semibold">All files</h2>

        {/* Table */}
        <div className="flex-1 overflow-auto rounded-lg border border-border/50">
          <table className="w-full">
            <thead className="bg-card/30 sticky text-foreground/70 top-0">
              <tr className="text-xs">
                <th className="text-left py-3 px-4 font-medium">File name</th>
                <th className="text-left py-3 px-4 font-medium">Collection</th>
                <th className="text-left py-3 px-4 font-medium">
                  Last modified
                </th>
                <th className="w-10"></th>
              </tr>
            </thead>
            <tbody>
              {(filteredDocs ?? []).map((doc: any) => {
                const Icon = typeIcons[doc?.type] || typeIcons.default;
                return (
                  <tr
                    key={doc?.id}
                    onClick={() => handleDocumentClick(doc.id)}
                    className={`border-t border-border/30 cursor-pointer transition-colors ${
                      selectedDocId === doc.id
                        ? "bg-foreground/10"
                        : "hover:bg-card/50"
                    }`}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-card rounded-md">
                          <Icon className="h-4 w-4 text-foreground" />
                        </div>
                        <div>
                          <div className="font-medium text-sm">{doc.title}</div>
                          <div className="text-xs text-foreground/70">
                            {doc.size} · {doc.type}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-sm text-foreground">
                        {doc.collection}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-xs text-foreground/80">
                      {doc.date}
                    </td>
                    <td className="py-3 px-4">
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </td>
                  </tr>
                );
              })}
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
