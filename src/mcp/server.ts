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

// MIRA mascot faces - owl-themed (wisdom & memory)
const MIRA_FACES = {
  search: "[◉‿◉]",      // happy searching
  recent: "[◉_◉]",      // neutral browsing
  init: "[◉▽◉]",        // greeting
  status: "[◉~◉]",      // checking
  error: "[◉!◉]",       // alert
  decisions: "[◉?◉]",   // questioning
};

// Tool parameter schemas
const SearchSchema = z.object({
  query: z.string().describe("Search query for finding conversations"),
  limit: z.number().optional().default(10).describe("Maximum number of results"),
  project_path: z.string().optional().describe("Optional: filter to specific project path"),
  compact: z.boolean().optional().default(true).describe("Return compact format (default true, ~79% smaller)"),
  days: z.number().optional().describe("Filter to sessions from last N days (hard cutoff)"),
  recency_bias: z.boolean().optional().default(true).describe("Apply time decay to boost recent results (default true). Set false for historical searches."),
});

const RecentSchema = z.object({
  limit: z.number().optional().default(10).describe("Maximum number of recent sessions"),
  days: z.number().optional().describe("Filter to sessions from last N days (e.g., 7 for last week)"),
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
    "Search conversation history by keywords. Uses semantic search (with remote storage) or FTS5 keyword search (local).",
    SearchSchema.shape,
    async ({ query, limit, project_path, compact, days, recency_bias }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("search", { query, limit, project_path, compact, days, recency_bias }) as any;
        const resultCount = result?.results?.length || result?.total || 0;
        return {
          content: [{ type: "text", text: `${MIRA_FACES.search} ${resultCount} results\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.search} Search error: ${error}` }],
        };
      }
    }
  );

  server.tool(
    "mira_recent",
    "Show recent conversation sessions grouped by project.",
    RecentSchema.shape,
    async ({ limit, days }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("recent", { limit, days }) as any;
        const sessionCount = result?.sessions?.length || 0;
        return {
          content: [{ type: "text", text: `${MIRA_FACES.recent} ${sessionCount} sessions\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.recent} Recent error: ${error}` }],
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
        const result = await callRpc("init", { project_path }) as any;
        const userName = result?.custodian?.name || "friend";
        return {
          content: [{ type: "text", text: `${MIRA_FACES.init} Hello ${userName}!\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.init} Init error: ${error}` }],
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
        const result = await callRpc("status", { project_path }) as any;
        const mode = result?.storage_health?.mode || "local";
        return {
          content: [{ type: "text", text: `${MIRA_FACES.status} ${mode} mode\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.status} Status error: ${error}` }],
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
        const result = await callRpc("error_lookup", { query, limit }) as any;
        const solutionCount = result?.solutions?.length || result?.total || 0;
        return {
          content: [{ type: "text", text: `${MIRA_FACES.error} ${solutionCount} solutions found\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.error} Error lookup error: ${error}` }],
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
        const result = await callRpc("decisions", { query, category, limit }) as any;
        const decisionCount = result?.decisions?.length || result?.total || 0;
        return {
          content: [{ type: "text", text: `${MIRA_FACES.decisions} ${decisionCount} decisions\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.decisions} Decisions error: ${error}` }],
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
