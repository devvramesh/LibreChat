// mcp/web-search/index.js
import express from "express";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { z } from "zod";

const SEARXNG_URL = process.env.SEARXNG_URL || "http://searxng:8080";
const PORT = Number(process.env.PORT || 3002);

// Strip HTML tags from text
function stripHtml(html) {
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Search SearXNG
async function searchSearxng(query, limit = 10) {
  const url = new URL("/search", SEARXNG_URL);
  url.searchParams.set("q", query);
  url.searchParams.set("format", "json");

  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`SearXNG error ${res.status}: ${text}`);
  }

  const data = await res.json();
  const results = (data.results || []).slice(0, limit).map((r) => ({
    title: r.title,
    url: r.url,
    content: r.content || "",
    engine: r.engine,
  }));

  return {
    query: data.query,
    results,
    total: data.number_of_results || results.length,
  };
}

// Fetch and extract text from URL
async function fetchUrl(url, maxChars = 15000) {
  const res = await fetch(url, {
    headers: {
      "User-Agent": "Mozilla/5.0 (compatible; LibreChat-MCP/1.0)",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
    signal: AbortSignal.timeout(30000),
  });

  if (!res.ok) {
    throw new Error(`Fetch error ${res.status}: ${res.statusText}`);
  }

  const contentType = res.headers.get("content-type") || "";
  const html = await res.text();

  // If it's HTML, strip tags; otherwise return as-is
  let text = contentType.includes("text/html") ? stripHtml(html) : html;

  // Limit character count
  if (text.length > maxChars) {
    text = text.slice(0, maxChars) + "\n\n[Content truncated at " + maxChars + " characters]";
  }

  return text;
}

// Fetch with timeout for deep research (more aggressive HTML stripping)
async function fetchWithTimeout(url, maxChars, timeoutMs = 10000) {
  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) return null;

    const html = await response.text();
    const text = html
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
      .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
      .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
      .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/g, " ")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, maxChars);

    return text;
  } catch {
    return null;
  }
}

// Generate varied search queries from objective
function generateQueries(objective) {
  const base = objective.toLowerCase().replace(/[?.,!]/g, "");
  return [
    base,
    `${base} latest 2025 2026`,
    `${base} guide explained how`,
  ];
}

// Track active transports for message routing
const transports = new Map();

function createServer() {
  const server = new McpServer({
    name: "web-search",
    version: "1.0.0",
  });

  // Tool: web_search
  server.tool(
    "web_search",
    {
      query: z.string().min(1).describe("Search query"),
      limit: z.number().int().min(1).max(20).default(10).describe("Max results (default 10)"),
    },
    async ({ query, limit }) => {
      const data = await searchSearxng(query, limit);

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(data, null, 2),
          },
        ],
      };
    }
  );

  // Tool: web_fetch
  server.tool(
    "web_fetch",
    {
      url: z.string().url().describe("The URL to fetch"),
      max_chars: z.number().int().min(1000).max(50000).default(15000).describe("Max characters to return (default 15000, use lower values like 3000-5000 for summaries)"),
    },
    async ({ url, max_chars }) => {
      const text = await fetchUrl(url, max_chars);

      return {
        content: [
          {
            type: "text",
            text: text,
          },
        ],
      };
    }
  );

  // Tool: deep_research
  // Note: Using 3-argument form (name, schema, handler) to match SDK expectations
  server.tool(
    "deep_research",
    {
      objective: z.string().describe("The research question or objective. USE THIS TOOL when user says 'deep research', 'research thoroughly', 'investigate', or wants comprehensive analysis. Do NOT manually chain web_search/web_fetch."),
      max_searches: z.number().int().min(1).max(10).default(5).describe("Maximum search iterations (1-10, default 5)"),
      max_chars_per_source: z.number().int().min(1000).max(10000).default(4000).describe("Max chars to extract per source (default 4000)"),
    },
    async ({ objective, max_searches, max_chars_per_source }) => {
      console.log(`[deep_research] Called with objective: ${objective.substring(0, 50)}...`);
      const startTime = Date.now();
      const MAX_FETCHES = 10;
      const MAX_TOTAL_TIME = 60000; // 60 seconds
      const OUTPUT_CAP = 12000;

      const sources = [];
      const fetchedUrls = new Set();
      let queriesRun = 0;
      let totalChars = 0;

      // Generate search queries
      const queries = generateQueries(objective).slice(0, max_searches);

      // Search loop
      for (const query of queries) {
        if (Date.now() - startTime > MAX_TOTAL_TIME) break;
        if (fetchedUrls.size >= MAX_FETCHES) break;

        try {
          const searchResults = await searchSearxng(query, 5);
          queriesRun++;

          // Get new URLs (not already fetched)
          const newUrls = searchResults.results
            .filter((r) => !fetchedUrls.has(r.url))
            .slice(0, 3);

          // Fetch content from top 2-3 new URLs
          for (const result of newUrls) {
            if (Date.now() - startTime > MAX_TOTAL_TIME) break;
            if (fetchedUrls.size >= MAX_FETCHES) break;

            fetchedUrls.add(result.url);

            const content = await fetchWithTimeout(result.url, max_chars_per_source, 10000);
            if (content && content.length > 100) {
              sources.push({
                url: result.url,
                title: result.title,
                content: content,
              });
              totalChars += content.length;
            }
          }
        } catch (err) {
          // Log and continue on search errors
          console.error(`Search error for "${query}":`, err.message);
        }
      }

      // Compile report
      let report = `## Research Report: ${objective}\n\n`;
      report += `### Research Metadata\n`;
      report += `- Queries run: ${queriesRun}\n`;
      report += `- Sources fetched: ${sources.length}\n`;
      report += `- Total content analyzed: ${totalChars} chars\n`;
      report += `- Time taken: ${((Date.now() - startTime) / 1000).toFixed(1)}s\n\n`;

      report += `### Sources\n`;
      sources.forEach((s, i) => {
        report += `${i + 1}. [${s.title}](${s.url})\n`;
      });
      report += `\n`;

      report += `### Detailed Notes\n\n`;
      sources.forEach((s, i) => {
        report += `**Source ${i + 1}: ${s.title}**\n`;
        report += `URL: ${s.url}\n\n`;
        report += `${s.content.slice(0, 1500)}${s.content.length > 1500 ? "..." : ""}\n\n`;
        report += `---\n\n`;
      });

      // Cap output
      if (report.length > OUTPUT_CAP) {
        report = report.slice(0, OUTPUT_CAP) + "\n\n[Report truncated at " + OUTPUT_CAP + " characters]";
      }

      return {
        content: [
          {
            type: "text",
            text: report,
          },
        ],
      };
    }
  );

  console.log("[MCP Web Search] Registered tools: web_search, web_fetch, deep_research");
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
  console.log(`MCP web-search (SSE) listening on http://0.0.0.0:${PORT}/sse`);
});
