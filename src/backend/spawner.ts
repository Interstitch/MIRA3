/**
 * Backend Spawner - Manages the Python backend process
 * Handles spawning, communication via JSON-RPC, and shutdown
 */

import { spawn, ChildProcess } from "child_process";
import { createInterface, Interface } from "readline";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { existsSync } from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let backendProcess: ChildProcess | null = null;
let readline: Interface | null = null;
let requestId = 0;
const pendingRequests = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();

/**
 * Find the Python backend script
 */
function findBackendScript(): string {
  // Try multiple paths - development and production layouts
  const paths = [
    join(__dirname, "..", "..", "python", "mira_backend.py"),  // From dist/backend/
    join(__dirname, "..", "python", "mira_backend.py"),        // Alternative layout
    join(process.cwd(), "python", "mira_backend.py"),          // From CWD
  ];

  for (const p of paths) {
    if (existsSync(p)) {
      return p;
    }
  }

  throw new Error(`Python backend script not found. Tried: ${paths.join(", ")}`);
}

/**
 * Find Python executable
 */
function findPython(): string {
  // Try python3 first (Unix), then python (Windows)
  return process.platform === "win32" ? "python" : "python3";
}

/**
 * Spawn the Python backend process
 */
export async function spawnBackend(): Promise<void> {
  if (backendProcess) {
    return; // Already running
  }

  const pythonPath = findPython();
  const scriptPath = findBackendScript();

  return new Promise((resolve, reject) => {
    backendProcess = spawn(pythonPath, [scriptPath], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    if (!backendProcess.stdout || !backendProcess.stdin) {
      reject(new Error("Failed to create backend process pipes"));
      return;
    }

    readline = createInterface({
      input: backendProcess.stdout,
      crlfDelay: Infinity,
    });

    // Handle responses from Python backend
    readline.on("line", (line) => {
      try {
        const response = JSON.parse(line);
        if (response.id !== undefined && pendingRequests.has(response.id)) {
          const pending = pendingRequests.get(response.id)!;
          pendingRequests.delete(response.id);

          if (response.error) {
            pending.reject(new Error(response.error.message || "RPC error"));
          } else {
            pending.resolve(response.result);
          }
        } else if (response.method === "ready") {
          // Backend is ready
          resolve();
        }
      } catch {
        // Not JSON, might be debug output - ignore
      }
    });

    // Handle stderr (for debugging)
    backendProcess.stderr?.on("data", (data) => {
      console.error(`[Python Backend] ${data.toString()}`);
    });

    // Handle process exit
    backendProcess.on("exit", (code) => {
      console.error(`Python backend exited with code ${code}`);
      backendProcess = null;
      readline = null;

      // Reject all pending requests
      for (const [id, pending] of pendingRequests) {
        pending.reject(new Error("Backend process exited"));
        pendingRequests.delete(id);
      }
    });

    backendProcess.on("error", (error) => {
      reject(new Error(`Failed to spawn Python backend: ${error.message}`));
    });

    // Set a long timeout for backend startup (first run installs deps + downloads model)
    const startupTimeout = setTimeout(() => {
      reject(new Error("Backend startup timeout - first run may take several minutes"));
    }, 600000); // 10 minutes for first run

    // Clear timeout when ready signal is received (handled in readline "line" event)
    const originalResolve = resolve;
    resolve = () => {
      clearTimeout(startupTimeout);
      originalResolve();
    };
  });
}

/**
 * Call an RPC method on the Python backend
 */
export async function callRpc(method: string, params: Record<string, unknown>): Promise<unknown> {
  if (!backendProcess || !backendProcess.stdin) {
    throw new Error("Backend process not running");
  }

  const id = ++requestId;
  const request = {
    jsonrpc: "2.0",
    id,
    method,
    params,
  };

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { resolve, reject });

    try {
      backendProcess!.stdin!.write(JSON.stringify(request) + "\n");
    } catch (error) {
      pendingRequests.delete(id);
      reject(error);
    }

    // Timeout after 5 minutes (first-run may need to install deps + download model)
    const timeout = 300000;
    setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        reject(new Error("RPC timeout"));
      }
    }, timeout);
  });
}

/**
 * Shutdown the Python backend
 */
export async function shutdownBackend(): Promise<void> {
  if (backendProcess) {
    try {
      await callRpc("shutdown", {});
    } catch {
      // Ignore errors during shutdown
    }

    backendProcess.kill();
    backendProcess = null;
    readline = null;
  }
}
