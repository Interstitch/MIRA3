#!/usr/bin/env node

/**
 * MIRA3 - Memory Information Retriever and Archiver
 * CLI entry point for the MCP server
 */

import { startServer } from "./mcp/server.js";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { homedir } from "os";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Marker to identify MIRA SessionStart hooks in command
const HOOK_MARKER = "claude-mira3 --init";

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
  claude-mira3 --setup                  Manually configure SessionStart hook
  claude-mira3 --help                   Show this help message
  claude-mira3 --version                Show version

Init Mode:
  The --init flag runs mira_init directly and outputs JSON to stdout.
  This is designed for use in Claude Code SessionStart hooks.
  The hook runs on all session starts (startup, resume, clear, compact).

Auto-Setup:
  MIRA automatically configures the SessionStart hook on first MCP start.
  To disable auto-setup: touch ~/.claude/.mira-no-auto-hook
  To force re-setup: rm ~/.claude/.mira-hook-configured

The server automatically installs Python dependencies on first run into .mira/.venv/

Add to Claude Code:
  claude mcp add mira3 -- npx claude-mira3
`);
  process.exit(0);
}

if (args.includes("--version") || args.includes("-v")) {
  // Read version from package.json
  const pkgPath = join(__dirname, "..", "package.json");
  const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
  console.log(`claude-mira3 v${pkg.version}`);
  process.exit(0);
}

// Check for --setup mode (explicit hook configuration)
if (args.includes("--setup")) {
  const configured = ensureSessionStartHook(true);
  if (configured) {
    console.log("[MIRA] ✓ SessionStart hook configured/fixed successfully");
    console.log("[MIRA]   Restart Claude Code to activate");
  } else {
    console.log("[MIRA] SessionStart hook already configured correctly");
  }
  process.exit(0);
}

// Check for --init mode
if (args.includes("--init")) {
  runInitMode();
} else {
  // Auto-setup hook on first MCP server start (silent if already done)
  try {
    ensureSessionStartHook(false);
  } catch (e) {
    // Don't fail MCP startup over hook config
    console.error(`[MIRA] Could not auto-configure hook: ${(e as Error).message}`);
  }

  // Start the MCP server
  startServer();
}

/**
 * Ensure SessionStart hook is configured in Claude settings.
 *
 * @param force - If true, skip marker check and always attempt setup
 * @returns true if hook was newly configured, false if already exists
 */
function ensureSessionStartHook(force: boolean = false): boolean {
  const claudeDir = join(homedir(), ".claude");
  const settingsPath = join(claudeDir, "settings.json");
  const markerPath = join(claudeDir, ".mira-hook-configured");
  const disablePath = join(claudeDir, ".mira-no-auto-hook");

  // Check for explicit disable marker
  if (existsSync(disablePath) && !force) {
    return false;
  }

  // Skip if marker exists (already attempted setup) unless forced
  if (existsSync(markerPath) && !force) {
    return false;
  }

  try {
    let settings: Record<string, unknown> = {};

    if (existsSync(settingsPath)) {
      const content = readFileSync(settingsPath, "utf8");
      settings = JSON.parse(content);
    }

    // Check if hook already exists
    const settingsStr = JSON.stringify(settings);
    const hasHook = settingsStr.includes(HOOK_MARKER);

    if (hasHook) {
      // Check for old format with matcher (needs fixing)
      const sessionStartHooks = (settings.hooks as Record<string, unknown[]>)?.SessionStart || [];
      let needsFix = false;

      for (let i = 0; i < sessionStartHooks.length; i++) {
        const hook = sessionStartHooks[i] as Record<string, unknown>;
        if (hook.matcher && JSON.stringify(hook).includes(HOOK_MARKER)) {
          // Found old format - remove matcher
          delete hook.matcher;
          needsFix = true;
          console.error("[MIRA] Fixed SessionStart hook format (removed invalid matcher)");
        }
      }

      if (needsFix) {
        writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + "\n");
      }

      // Mark as configured so we don't check again
      mkdirSync(claudeDir, { recursive: true });
      writeFileSync(markerPath, new Date().toISOString());
      return needsFix;  // Return true if we fixed something
    }

    // Initialize hooks structure
    if (!settings.hooks) settings.hooks = {};
    const hooks = settings.hooks as Record<string, unknown[]>;
    if (!hooks.SessionStart) hooks.SessionStart = [];

    // Add MIRA hook
    // Note: SessionStart hooks don't use matchers - the source field in input
    // indicates startup/resume/clear/compact. We run on all session starts.
    hooks.SessionStart.push({
      hooks: [{
        type: "command",
        command: `npx claude-mira3 --init --project="$CLAUDE_PROJECT_DIR" --quiet 2>/dev/null || echo '{"guidance":{"actions":["MIRA unavailable"]}}'`,
        timeout: 30000
      }]
    });

    // Write back with formatting
    mkdirSync(claudeDir, { recursive: true });
    writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + "\n");

    // Mark as configured
    writeFileSync(markerPath, new Date().toISOString());

    // Log to stderr (not stdout, which is for MCP protocol)
    // Only log in auto-setup mode, not when called via --setup
    if (!force) {
      console.error("[MIRA] ✓ Auto-configured SessionStart hook");
      console.error("[MIRA]   Future sessions will receive MIRA context automatically");
      console.error("[MIRA]   Restart Claude Code to activate");
      console.error("[MIRA]   To disable: touch ~/.claude/.mira-no-auto-hook");
    }

    return true;

  } catch (e) {
    // Don't fail on config errors
    if (force) {
      throw e; // Re-throw in explicit --setup mode
    }
    return false;
  }
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
