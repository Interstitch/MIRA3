/**
 * Integration test - verify Node.js → Python backend communication
 */

import { spawnBackend, callRpc, shutdownBackend } from "../src/backend/spawner.js";

async function runTests() {
  console.log("=== MIRA3 Integration Tests ===\n");

  try {
    // Test 1: Spawn backend
    console.log("1. Spawning Python backend...");
    await spawnBackend();
    console.log("   ✓ Backend spawned successfully\n");

    // Test 2: Call status
    console.log("2. Calling status RPC...");
    const status = await callRpc("status", {});
    console.log("   ✓ Status response:", JSON.stringify(status, null, 2), "\n");

    // Test 3: Call search
    console.log("3. Calling search RPC...");
    const searchResult = await callRpc("search", { query: "authentication", limit: 5 });
    console.log("   ✓ Search response:", JSON.stringify(searchResult, null, 2), "\n");

    // Test 4: Call recent
    console.log("4. Calling recent RPC...");
    const recentResult = await callRpc("recent", { limit: 5 });
    console.log("   ✓ Recent response:", JSON.stringify(recentResult, null, 2), "\n");

    // Test 5: Call init
    console.log("5. Calling init RPC...");
    const initResult = await callRpc("init", { project_path: "/workspaces/MIRA3" });
    console.log("   ✓ Init response:", JSON.stringify(initResult, null, 2), "\n");

    console.log("=== All tests passed! ===");

  } catch (error) {
    console.error("Test failed:", error);
    process.exit(1);
  } finally {
    // Shutdown
    console.log("\nShutting down backend...");
    await shutdownBackend();
    console.log("Done.");
  }
}

runTests();
