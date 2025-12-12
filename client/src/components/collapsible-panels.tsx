"use client";

import * as React from "react";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { cn } from "@/lib/utils";
import type { ImperativePanelHandle } from "react-resizable-panels";

interface PanelControlContextType {
  openPanel: () => void;
  closePanel: () => void;
}

const PanelControlContext = React.createContext<PanelControlContextType | null>(
  null
);

export function usePanelControl() {
  const context = React.useContext(PanelControlContext);
  if (!context) {
    throw new Error("usePanelControl must be used within CollapsiblePanels");
  }
  return context;
}

interface CollapsiblePanelsProps {
  // Content for the left panel
  leftPanel: React.ReactNode;

  // Content for the right panel
  rightPanel: React.ReactNode;

  // Size of the right panel when open (percentage) (default 40)
  rightPanelSize?: number;

  // Minimum size for the right panel (percentage) (default 20)
  minSize?: number;

  // Maximum size for the right panel (percentage) (default 100)
  maxSize?: number;

  // Additional className for the container
  className?: string;

  // Whether to show handle indicator (default true)
  showHandle?: boolean;
}

/**
 * CollapsiblePanels component with a collapsible right panel.
 * Right panel starts closed (0%) and can be opened programmatically.
 * Use usePanelControl() hook in child components to control the panel.
 */
export function CollapsiblePanels({
  leftPanel,
  rightPanel,
  rightPanelSize = 40,
  minSize = 20,
  maxSize = 100,
  className,
  showHandle = true,
}: CollapsiblePanelsProps) {
  const rightPanelRef = React.useRef<ImperativePanelHandle>(null);

  const openPanel = React.useCallback(() => {
    rightPanelRef.current?.resize(rightPanelSize);
  }, [rightPanelSize]);

  const closePanel = React.useCallback(() => {
    rightPanelRef.current?.resize(0);
  }, []);

  const contextValue = React.useMemo<PanelControlContextType>(
    () => ({
      openPanel,
      closePanel,
    }),
    [openPanel, closePanel]
  );

  return (
    <PanelControlContext.Provider value={contextValue}>
      <ResizablePanelGroup
        direction="horizontal"
        className={cn("h-full w-full", className)}
      >
        <ResizablePanel defaultSize={100} minSize={50} className="min-w-0">
          {leftPanel}
        </ResizablePanel>

        <ResizableHandle withHandle={showHandle} />

        <ResizablePanel
          ref={rightPanelRef}
          defaultSize={0}
          minSize={minSize}
          maxSize={maxSize}
          collapsible={true}
          className="min-w-0"
        >
          {rightPanel}
        </ResizablePanel>
      </ResizablePanelGroup>
    </PanelControlContext.Provider>
  );
}
