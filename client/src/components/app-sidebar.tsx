"use client";

import * as React from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import {
  Plus,
  Search,
  Package,
  FileText,
  BarChart3,
  Circle,
  Loader2,
} from "lucide-react";
import { useDocuments } from "./panel-context";

export function AppSidebar() {
  const { isConnected, isConnecting, error, connect } = useDocuments();
  const [activeView, setActiveView] = React.useState("documents");

  return (
    <Sidebar className="bg-background border-border px-2">
      <SidebarHeader className="p-6 border-border">
        <div className="flex items-center gap-2 mb-5">
          <h2 className="text-lg font-semibold">MCP Doc Server</h2>
        </div>
        <Button className="w-full py-5 bg-primary text-primary-foreground hover:bg-primary/90">
          <div className="mr-2 flex h-5 w-5 items-center justify-center rounded-full bg-[#060606]">
            <Plus className="h-3 w-3 text-white" />
          </div>
          Index Document
        </Button>
      </SidebarHeader>

      <SidebarContent className="p-4">
        <SidebarMenu className="gap-y-2">
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => setActiveView("search")}
              isActive={activeView === "search"}
              className="w-full justify-start cursor-pointer hover:bg-card hover:text-white"
            >
              <Search className="h-4 w-4" />
              <span>Search</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => setActiveView("collections")}
              isActive={activeView === "collections"}
              className="w-full justify-start cursor-pointer hover:bg-card hover:text-white"
            >
              <Package className="h-4 w-4" />
              <span>Collections</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => setActiveView("documents")}
              isActive={activeView === "documents"}
              className="w-full justify-start cursor-pointer hover:bg-card hover:text-white"
            >
              <FileText className="h-4 w-4" />
              <span>Documents</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => setActiveView("stats")}
              isActive={activeView === "stats"}
              className="w-full justify-start cursor-pointer hover:bg-card hover:text-white"
            >
              <BarChart3 className="h-4 w-4" />
              <span>Statistics</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarContent>

      <SidebarFooter className="p-4">
        {/* Server Status */}
        <div className="p-4">
          {isConnecting ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-3 w-3 animate-spin text-yellow-500" />
              <span className="text-xs text-foreground/70">Connecting...</span>
            </div>
          ) : isConnected ? (
            <div className="flex items-center gap-2">
              <Circle className="h-2 w-2 fill-primary text-primary" />
              <span className="text-xs font-bold text-foreground">
                Server Connected
              </span>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <Circle className="h-2 w-2 fill-red-500 text-red-500" />
                <span className="text-xs text-red-400">Disconnected</span>
              </div>
              {error && (
                <p className="text-xs text-red-400/70 truncate" title={error}>
                  {error}
                </p>
              )}
              <button
                onClick={() => connect()}
                className="text-xs text-primary hover:underline text-left"
              >
                Retry connection
              </button>
            </div>
          )}
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
