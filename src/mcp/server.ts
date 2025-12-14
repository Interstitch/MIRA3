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
  init: "[◉▽◉]",        // greeting (first - session start)
  search: "[◉‿◉]",      // happy searching
  recent: "[◉_◉]",      // neutral browsing
  error: "[◉!◉]",       // alert
  decisions: "[◉?◉]",   // questioning
  code: "[◉⌘◉]",        // code archaeology
  status: "[◉~◉]",      // checking (last - admin)
};

// Tool parameter schemas
const InitSchema = z.object({
  project_path: z.string().optional().describe("Current project path for context"),
});

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

const ErrorLookupSchema = z.object({
  query: z.string().describe("Error message or description to search for"),
  limit: z.number().optional().default(5).describe("Maximum number of results"),
});

const DecisionsSchema = z.object({
  query: z.string().describe("Search query for finding decisions"),
  category: z.string().optional().describe("Optional category filter: architecture, technology, implementation, testing, security, performance, workflow"),
  limit: z.number().optional().default(10).describe("Maximum number of results"),
  min_confidence: z.number().optional().default(0).describe("Minimum confidence threshold (0.0-1.0). Use 0.8+ for explicit decisions only, 0.6+ to include implicit."),
});

const CodeHistorySchema = z.object({
  path: z.string().optional().describe("File path or pattern (e.g., 'handlers.py', 'src/*.py'). Use * for wildcards."),
  symbol: z.string().optional().describe("Function/class/method name to search (e.g., 'handle_search', 'MyClass')"),
  mode: z.enum(["timeline", "snapshot", "changes"]).optional().default("timeline").describe("timeline: list changes over time, snapshot: reconstruct file at date, changes: show edit details"),
  date: z.string().optional().describe("Target date for snapshot mode (ISO format, e.g., '2025-12-01')"),
  limit: z.number().optional().default(20).describe("Maximum results"),
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

  // ==================== Register tools (ordered by typical usage) ====================

  // 1. MIRA_INIT - Session start, context setup
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
        const userName = result?.core?.custodian?.name || result?.custodian?.name || "friend";
        const sessionCount = result?.core?.custodian?.total_sessions || 0;
        const storageMode = result?.storage?.mode || "local";

        // Build human-readable summary
        let summary = `${MIRA_FACES.init} Hello ${userName}!`;
        summary += `\n  ${sessionCount} session${sessionCount !== 1 ? 's' : ''} of shared history`;
        summary += ` (${storageMode === "central" ? "central sync enabled" : "local mode"})`;

        // Show indexing status if relevant
        if (result?.indexing?.pending > 0) {
          summary += `\n  ${result.indexing.indexed}/${result.indexing.total} sessions indexed (${result.indexing.pending} pending)`;
        }

        // Show active ingestion if running
        if (result?.active_ingestion?.count > 0) {
          summary += `\n  ⏳ ${result.active_ingestion.count} conversation${result.active_ingestion.count !== 1 ? 's' : ''} currently being indexed`;
        }

        // Show high-priority alerts
        const highAlerts = (result?.alerts || []).filter((a: any) => a.priority === "high");
        if (highAlerts.length > 0) {
          summary += `\n  ⚠️ ${highAlerts.length} alert${highAlerts.length !== 1 ? 's' : ''} requiring attention`;
        }

        // Show token estimate
        if (result?.token_estimate?.tokens) {
          summary += `\n  📊 ~${result.token_estimate.tokens.toLocaleString()} tokens injected into context`;
        }

        return {
          content: [{ type: "text", text: `${summary}\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.init} Init error: ${error}` }],
        };
      }
    }
  );

  // 2. MIRA_SEARCH - Primary search function
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

        // Build human-readable summary
        let summary = `${MIRA_FACES.search} Found ${resultCount} result${resultCount !== 1 ? 's' : ''} for "${query}"`;

        if (resultCount > 0) {
          // Show top result summaries
          const topResults = (result?.results || []).slice(0, 3);
          const previews = topResults
            .map((r: any) => r.summary || r.s || "No summary")
            .filter((s: string) => s && s !== "No summary")
            .slice(0, 2);
          if (previews.length > 0) {
            summary += `\n  • ${previews.join('\n  • ')}`;
          }
        } else if (result?.message) {
          summary += `\n  ${result.message}`;
        }

        if (result?.corrections?.length > 0) {
          summary += `\n  [Typo corrected: ${result.original_query} → ${query}]`;
        }

        return {
          content: [{ type: "text", text: `${summary}\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.search} Search error: ${error}` }],
        };
      }
    }
  );

  // 3. MIRA_RECENT - Browse recent sessions
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
        const totalSessions = result?.total || 0;
        const projects = result?.projects || [];

        // Build human-readable summary
        let summary = `${MIRA_FACES.recent} ${totalSessions} recent session${totalSessions !== 1 ? 's' : ''}`;
        if (days) {
          summary += ` from last ${days} day${days !== 1 ? 's' : ''}`;
        }
        summary += ` across ${projects.length} project${projects.length !== 1 ? 's' : ''}`;

        // Show recent session previews
        if (totalSessions > 0 && projects.length > 0) {
          const recentSessions = projects.flatMap((p: any) => p.sessions || []).slice(0, 3);
          const previews = recentSessions
            .map((s: any) => s.summary || "No summary")
            .filter((s: string) => s && s !== "No summary")
            .slice(0, 2);
          if (previews.length > 0) {
            summary += `\n  • ${previews.join('\n  • ')}`;
          }
        } else if (totalSessions === 0) {
          summary += `\n  No sessions found. MIRA may still be indexing conversation history.`;
        }

        return {
          content: [{ type: "text", text: `${summary}\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.recent} Recent error: ${error}` }],
        };
      }
    }
  );

  // 4. MIRA_ERROR_LOOKUP - Specialized error search
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
        const solutions = result?.solutions || [];
        const solutionCount = solutions.length || result?.total || 0;

        // Build human-readable summary
        let summary = `${MIRA_FACES.error} Found ${solutionCount} past solution${solutionCount !== 1 ? 's' : ''} for similar errors`;

        if (solutionCount > 0) {
          // Show top solutions
          const topSolutions = solutions.slice(0, 2);
          const previews = topSolutions.map((s: any) => {
            const errorType = s.error_type || "Error";
            const solution = s.solution ? s.solution.substring(0, 80) + "..." : "See details";
            return `${errorType}: ${solution}`;
          });
          if (previews.length > 0) {
            summary += `\n  • ${previews.join('\n  • ')}`;
          }
        } else {
          summary += `\n  No past solutions found. This may be a new error pattern.`;
        }

        return {
          content: [{ type: "text", text: `${summary}\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.error} Error lookup error: ${error}` }],
        };
      }
    }
  );

  // 5. MIRA_DECISIONS - Specialized decision search
  server.tool(
    "mira_decisions",
    "Search for past architectural and design decisions with their reasoning.",
    DecisionsSchema.shape,
    async ({ query, category, limit, min_confidence }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("decisions", { query, category, limit, min_confidence }) as any;
        const decisions = result?.decisions || [];
        const decisionCount = decisions.length || result?.total || 0;

        // Build human-readable summary
        let summary = `${MIRA_FACES.decisions} Found ${decisionCount} past decision${decisionCount !== 1 ? 's' : ''}`;
        if (category) {
          summary += ` in category "${category}"`;
        }
        summary += ` matching "${query}"`;

        if (decisionCount > 0) {
          // Show top decisions
          const topDecisions = decisions.slice(0, 2);
          const previews = topDecisions.map((d: any) => {
            const text = d.decision || d.text || "No description";
            return text.length > 80 ? text.substring(0, 80) + "..." : text;
          });
          if (previews.length > 0) {
            summary += `\n  • ${previews.join('\n  • ')}`;
          }
        } else {
          summary += `\n  No past decisions found. Try a broader search term.`;
        }

        return {
          content: [{ type: "text", text: `${summary}\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.decisions} Decisions error: ${error}` }],
        };
      }
    }
  );

  // 6. MIRA_CODE_HISTORY - Code archaeology
  server.tool(
    "mira_code_history",
    "Search code history by file path or symbol name. Shows timeline of changes across sessions, can reconstruct file content at any historical date.",
    CodeHistorySchema.shape,
    async ({ path, symbol, mode, date, limit }) => {
      if (!backendReady) {
        return {
          content: [{ type: "text", text: "Error: Python backend not ready. Ensure Python 3.8+ is installed." }],
        };
      }
      try {
        const result = await callRpc("code_history", { path, symbol, mode, date, limit }) as any;
        const total = result?.total || 0;
        const modeLabel = result?.mode || mode || "timeline";

        // Build human-readable summary based on mode
        let summary = `${MIRA_FACES.code} Code History`;
        const searchTarget = symbol ? `symbol "${symbol}"` : (path ? `file "${path}"` : "all files");

        if (modeLabel === "timeline") {
          summary += ` - ${total} change${total !== 1 ? 's' : ''} to ${searchTarget}`;
          if (total > 0 && result?.timeline) {
            const firstChange = result.timeline[0];
            if (firstChange?.date) {
              summary += `\n  Most recent: ${firstChange.date.substring(0, 10)}`;
            }
          }
        } else if (modeLabel === "snapshot") {
          const confidence = result?.confidence || 0;
          const lineCount = result?.line_count || 0;
          summary += ` - Reconstructed ${searchTarget}`;
          if (date) summary += ` as of ${date}`;
          summary += `\n  ${lineCount} lines, ${Math.round(confidence * 100)}% confidence`;
        } else if (modeLabel === "changes") {
          summary += ` - ${total} edit${total !== 1 ? 's' : ''} to ${searchTarget}`;
        }

        if (total === 0 && !result?.content) {
          summary += `\n  No history found. This file may not have been read/edited in tracked sessions.`;
        }

        return {
          content: [{ type: "text", text: `${summary}\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.code} Code history error: ${error}` }],
        };
      }
    }
  );

  // 7. MIRA_STATUS - System admin (least common)
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
        const sessionCount = result?.global?.ingestion?.total_in_db || result?.sessions?.total || 0;
        const artifactCount = result?.global?.artifacts?.total || result?.artifacts?.total || 0;

        // Build human-readable summary
        let summary = `${MIRA_FACES.status} MIRA Status - ${mode === "central" ? "Central Storage" : "Local Mode"}`;

        // Session and artifact counts
        summary += `\n  ${sessionCount} session${sessionCount !== 1 ? 's' : ''} indexed, ${artifactCount} artifact${artifactCount !== 1 ? 's' : ''} stored`;

        // Sync status if applicable
        if (result?.sync_queue?.total_pending > 0) {
          summary += `\n  ${result.sync_queue.total_pending} item${result.sync_queue.total_pending !== 1 ? 's' : ''} pending sync to central`;
        }

        // Health indicators
        const health = result?.storage_health;
        if (health) {
          if (health.local_healthy === false) {
            summary += `\n  ⚠️ Local storage issue detected`;
          }
          if (health.central_configured && !health.central_available) {
            summary += `\n  ⚠️ Central storage configured but unreachable`;
          }
          // Security warnings
          if (health.security_warnings?.length > 0) {
            summary += `\n  🔓 Security: ${health.security_warnings[0].message}`;
          }
        }

        return {
          content: [{ type: "text", text: `${summary}\n${JSON.stringify(result, null, 2)}` }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `${MIRA_FACES.status} Status error: ${error}` }],
        };
      }
    }
  );

  // Handle graceful shutdown
  process.on("SIGINT", async () => {
    console.error("[MIRA MCP] Received SIGINT, shutting down backend");
    await shutdownBackend();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    console.error("[MIRA MCP] Received SIGTERM, shutting down backend");
    await shutdownBackend();
    process.exit(0);
  });

  // Connect to stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);
}
