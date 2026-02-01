// mcp/chat-search/index.js
import express from "express";
import { URL } from "node:url";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { z } from "zod";

const MEILI_URL = process.env.MEILI_URL || "http://chat-meilisearch:7700";
const MEILI_KEY = process.env.MEILI_MASTER_KEY; // master key or search key
const PORT = Number(process.env.PORT || 3001);

if (!MEILI_KEY) {
  console.error("Missing MEILI_MASTER_KEY env var");
  process.exit(1);
}

async function meiliSearchMessages(query, limit, userOnly = true) {
  const u = new URL("/indexes/messages/search", MEILI_URL);
  
  const searchBody = { q: query, limit };
  
  // Filter to only return user messages (not assistant responses)
  // This prevents the model from finding its own hallucinated responses
  if (userOnly) {
    searchBody.filter = 'sender = "User"';
  }
  
  const res = await fetch(u, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${MEILI_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(searchBody),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Meili error ${res.status}: ${text}`);
  }
  return res.json();
}

// Track active transports for message routing
const transports = new Map();

function createServer() {
  const server = new McpServer({
    name: "chat-search",
    version: "1.0.0",
  });

  // Tool: search_chat_history
  server.tool(
    "search_chat_history",
    {
      query: z.string().min(1).describe("What to search for in prior chats"),
      limit: z.number().int().min(1).max(20).default(8)
        .describe("Max results"),
    },
    async ({ query, limit }) => {
      const data = await meiliSearchMessages(query, limit);

      // Return compact, model-friendly output
      const hits = (data.hits || []).map((h) => ({
        conversationId: h.conversationId,
        messageId: h.messageId,
        sender: h.sender,
        text: h.text,
      }));

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                query: data.query,
                total: data.estimatedTotalHits ?? hits.length,
                hits,
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  return server;
}

const app = express();

// SSE endpoint - client connects here for server-sent events
app.get("/sse", async (req, res) => {
  console.log("SSE connection established");
  
  const transport = new SSEServerTransport("/messages", res);
  const server = createServer();
  
  transports.set(transport.sessionId, { transport, server });
  
  // Send keepalive ping every 15 seconds to prevent timeout
  const keepalive = setInterval(() => {
    if (!res.writableEnded) {
      res.write(":ping\n\n");
    }
  }, 15000);
  
  res.on("close", () => {
    console.log("SSE connection closed");
    clearInterval(keepalive);
    transports.delete(transport.sessionId);
    server.close().catch(() => {});
  });
  
  await server.connect(transport);
});

// Messages endpoint - client sends messages here
// IMPORTANT: Do NOT use express.json() - the SDK needs the raw request stream
app.post("/messages", async (req, res) => {
  const sessionId = req.query.sessionId;
  const session = transports.get(sessionId);
  
  if (!session) {
    res.status(400).json({ error: "Invalid or missing sessionId" });
    return;
  }
  
  await session.transport.handlePostMessage(req, res);
});

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`MCP chat-search (SSE) listening on http://0.0.0.0:${PORT}/sse`);
});
