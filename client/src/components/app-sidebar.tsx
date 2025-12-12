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
} from "lucide-react";

interface AppSidebarProps {
  activeView: string;
  onViewChangeAction: (view: string) => void;
}

export function AppSidebar({
  activeView,
  onViewChangeAction,
}: AppSidebarProps) {
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
              onClick={() => onViewChangeAction("search")}
              isActive={activeView === "search"}
              className="w-full justify-start cursor-pointer hover:bg-card hover:text-white"
            >
              <Search className="h-4 w-4" />
              <span>Search</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => onViewChangeAction("collections")}
              isActive={activeView === "collections"}
              className="w-full justify-start cursor-pointer hover:bg-card hover:text-white"
            >
              <Package className="h-4 w-4" />
              <span>Collections</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => onViewChangeAction("documents")}
              isActive={activeView === "documents"}
              className="w-full justify-start cursor-pointer hover:bg-card hover:text-white"
            >
              <FileText className="h-4 w-4" />
              <span>Documents</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => onViewChangeAction("stats")}
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
          <div className="flex items-center gap-2">
            <Circle className="h-2 w-2 fill-primary text-primary animate-pulse" />
            <span className="text-xs font-bold text-foreground">
              Server Connected
            </span>
            <span className="text-xs text-foreground ">(v1.0.0)</span>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
