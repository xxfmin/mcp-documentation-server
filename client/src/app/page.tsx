"use client";

import { CollapsiblePanels } from "@/components/collapsible-panels";
import { LeftPanelContent } from "@/components/LeftPanelContent";
import { RightPanelContent } from "@/components/RightPanelContent";
import { DocumentProvider } from "@/components/panel-context";

export default function Home() {
  return (
    <DocumentProvider>
      <div className="h-full w-full">
        <CollapsiblePanels
          leftPanel={<LeftPanelContent />}
          rightPanel={<RightPanelContent />}
          rightPanelSize={30}
          minSize={25}
          maxSize={40}
        />
      </div>
    </DocumentProvider>
  );
}
