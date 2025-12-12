/**
 * MCP Server - Thin layer that handles MCP protocol and delegates to Python backend
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { spawnBackend, callRpc, shutdownBackend } from "../backend/spawner.js";
import { createRequire } from "module";

// Read version from package.json (single source of truth)
const require = createRequire(import.meta.url);
const pkg = require("../../package.json");
const VERSION = pkg.version;

// Tool parameter schemas
const SearchSchema = z.object({
  query: z.string().describe("Search query for finding conversations"),
  limit: z.number().optional().default(10).describe("Maximum number of results"),
  project_path: z.string().optional().describe("Optional: filter to specific project path"),
});

const RecentSchema = z.object({
  limit: z.number().optional().default(10).describe("Maximum number of recent sessions"),
});

const InitSchema = z.object({
  project_path: z.string().optional().describe("Current project path for context"),
});

const ErrorLookupSchema = z.object({
  query: z.string().describe("Error message or description to search for"),
  limit: z.number().optional().default(5).describe("Maximum number of results"),
});

const DecisionsSchema = z.object({
  query: z.string().describe("Search query for finding decisions"),
  category: z.string().optional().describe("Optional category filter: architecture, technology, implementation, testing, security, performance, workflow"),
  limit: z.number().optional().default(10).describe("Maximum number of results"),
});

const StatusSchema = z.object({
  project_path: z.string().optional().describe("Optional: filter statistics to a specific project path. Returns both project-specific and global stats."),
});

export async function startServer(): Promise<void> {
  const server = new McpServer({
    name: "claude-mira3",
    version: VERSION,
  });

  // Spawn Python backend on startup
  let backendReady = false;

  try {
    await spawnBackend();
    backendReady = true;
  } catch (error) {
    console.error("Failed to start Python backend:", error);
    // Continue - tools will return errors if backend isn't ready
  }

  // Register tools
  server.tool(
    "mira_search",
    "Search conversation history by keywords. Uses ChromaDB semantic search with cosine similarity.",
    SearchSchema.shape,
    async ({ query, limit, project_path }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("search", { query, limit, project_path });
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Search error: ${error}` }],
        };
      }
    }
  );

  server.tool(
    "mira_recent",
    "Show recent conversation sessions grouped by project.",
    RecentSchema.shape,
    async ({ limit }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("recent", { limit });
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Recent error: ${error}` }],
        };
      }
    }
  );

  server.tool(
    "mira_init",
    "Get context summary for the current project. Useful at session start.",
    InitSchema.shape,
    async ({ project_path }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("init", { project_path });
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Init error: ${error}` }],
        };
      }
    }
  );

  server.tool(
    "mira_status",
    "Show ingestion statistics and system health. Optionally filter by project path for project-specific stats.",
    StatusSchema.shape,
    async ({ project_path }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("status", { project_path });
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Status error: ${error}` }],
        };
      }
    }
  );

  server.tool(
    "mira_error_lookup",
    "Search for past error solutions. Find how similar errors were resolved in previous sessions.",
    ErrorLookupSchema.shape,
    async ({ query, limit }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("error_lookup", { query, limit });
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Error lookup error: ${error}` }],
        };
      }
    }
  );

  server.tool(
    "mira_decisions",
    "Search for past architectural and design decisions with their reasoning.",
    DecisionsSchema.shape,
    async ({ query, category, limit }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("decisions", { query, category, limit });
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Decisions error: ${error}` }],
        };
      }
    }
  );

  // Handle graceful shutdown
  process.on("SIGINT", async () => {
    await shutdownBackend();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    await shutdownBackend();
    process.exit(0);
  });

  // Connect to stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);
}
