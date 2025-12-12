#!/usr/bin/env node

/**
 * MIRA3 - Memory Information Retriever and Archiver
 * CLI entry point for the MCP server
 */

import { startServer } from "./mcp/server.js";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { existsSync } from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const args = process.argv.slice(2);

if (args.includes("--help") || args.includes("-h")) {
  console.log(`
MIRA3 - Memory Information Retriever and Archiver

An MCP server for Claude Code conversation history with semantic search.

Usage:
  claude-mira3                          Start the MCP server (default)
  claude-mira3 --init                   Run mira_init and output JSON (for hooks)
  claude-mira3 --init --project=PATH    Run init with project context
  claude-mira3 --init --quiet           Suppress log output (JSON only)
  claude-mira3 --help                   Show this help message
  claude-mira3 --version                Show version

Init Mode:
  The --init flag runs mira_init directly and outputs JSON to stdout.
  This is designed for use in Claude Code SessionStart hooks:

  {
    "hooks": {
      "SessionStart": [{
        "matcher": { "type": ["startup", "compact"] },
        "hooks": [{
          "type": "command",
          "command": "npx claude-mira3 --init --project=\\"$CLAUDE_PROJECT_DIR\\" --quiet"
        }]
      }]
    }
  }

The server automatically installs Python dependencies on first run into .mira/.venv/

Add to Claude Code:
  claude mcp add mira3 -- npx claude-mira3
`);
  process.exit(0);
}

if (args.includes("--version") || args.includes("-v")) {
  console.log("claude-mira3 v0.2.21");
  process.exit(0);
}

// Check for --init mode
if (args.includes("--init")) {
  runInitMode();
} else {
  // Start the MCP server
  startServer();
}

/**
 * Run init mode - spawns Python with --init flag, outputs JSON
 */
function runInitMode(): void {
  const pythonArgs = ["--init"];

  // Extract --project argument
  const projectArg = args.find(a => a.startsWith("--project="));
  if (projectArg) {
    pythonArgs.push(projectArg);
  }

  // Extract --quiet flag
  if (args.includes("--quiet") || args.includes("-q")) {
    pythonArgs.push("--quiet");
  }

  // Find Python backend script
  const scriptPath = findBackendScript();
  const pythonPath = process.platform === "win32" ? "python" : "python3";

  const proc = spawn(pythonPath, [scriptPath, ...pythonArgs], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env },
  });

  // Pipe stdout (JSON output) to our stdout
  proc.stdout?.on("data", (data) => {
    process.stdout.write(data);
  });

  // Pipe stderr (logs) to our stderr
  proc.stderr?.on("data", (data) => {
    process.stderr.write(data);
  });

  proc.on("exit", (code) => {
    process.exit(code ?? 0);
  });

  proc.on("error", (error) => {
    console.error(`Failed to run init: ${error.message}`);
    process.exit(1);
  });
}

/**
 * Find the Python backend script
 */
function findBackendScript(): string {
  const paths = [
    join(__dirname, "..", "python", "mira_backend.py"),     // From dist/
    join(__dirname, "..", "..", "python", "mira_backend.py"), // From dist/backend/
    join(process.cwd(), "python", "mira_backend.py"),       // From CWD
  ];

  for (const p of paths) {
    if (existsSync(p)) {
      return p;
    }
  }

  throw new Error(`Python backend script not found. Tried: ${paths.join(", ")}`);
}
