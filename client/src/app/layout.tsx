import type { Metadata } from "next";
import { Gantari } from "next/font/google";
import "./globals.css";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";

const gantari = Gantari({
  variable: "--font-gantari",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "MCP Documentation Server",
  description: "Documentation server client",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${gantari.variable} font-sans antialiased`}>
        <SidebarProvider>
          <div className="flex h-screen w-full">
            <AppSidebar />
            <main className="flex-1 h-screen w-full bg-background p-2.5">
              <div className="h-full w-full rounded-md bg-card">{children}</div>
            </main>
          </div>
        </SidebarProvider>
      </body>
    </html>
  );
}
