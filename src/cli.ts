#!/usr/bin/env node

/**
 * MIRA3 - Memory Information Retriever and Archiver
 * CLI entry point for the MCP server
 */

import { startServer } from "./mcp/server.js";

const args = process.argv.slice(2);

if (args.includes("--help") || args.includes("-h")) {
  console.log(`
MIRA3 - Memory Information Retriever and Archiver

An MCP server for Claude Code conversation history with semantic search.

Usage:
  claude-mira3              Start the MCP server (default)
  claude-mira3 --help       Show this help message
  claude-mira3 --version    Show version

The server automatically installs Python dependencies (ChromaDB, sentence-transformers)
on first run into .mira/.venv/

Add to Claude Code:
  claude mcp add mira3 -- npx claude-mira3
`);
  process.exit(0);
}

if (args.includes("--version") || args.includes("-v")) {
  console.log("claude-mira3 v0.1.0");
  process.exit(0);
}

// Start the MCP server
startServer();
