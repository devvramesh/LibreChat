#!/usr/bin/env python3
"""
MCP Memory Server using Mem0 for intelligent memory extraction and semantic retrieval.
Uses Ollama for LLM and embeddings, Qdrant for vector storage.
"""

import os
import json
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from mem0 import Memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
PORT = int(os.environ.get("PORT", 3003))
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Mem0 configuration
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": LLM_MODEL,
            "ollama_base_url": OLLAMA_HOST,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": EMBED_MODEL,
            "ollama_base_url": OLLAMA_HOST,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
        }
    }
}

# Initialize Mem0 (lazy initialization)
memory: Optional[Memory] = None


def get_memory() -> Memory:
    """Get or initialize the Mem0 client."""
    global memory
    if memory is None:
        logger.info("Initializing Mem0 with config: %s", config)
        memory = Memory.from_config(config)
        logger.info("Mem0 initialized successfully")
    return memory


# Track active SSE connections
sessions = {}


def create_mcp_server() -> Server:
    """Create and configure the MCP server with memory tools."""
    server = Server("memory")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="add_memory",
                description="Store a new memory. Call this when the user shares important information about themselves, their preferences, projects, or anything worth remembering for future conversations. Do NOT store trivial or temporary information.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (default: 'default')",
                            "default": "default"
                        }
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="search_memory",
                description="Search memories semantically. Use this to recall information from past conversations. Returns relevant memories ranked by relevance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memories"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (default: 'default')",
                            "default": "default"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_all_memories",
                description="Get all stored memories for a user. Use sparingly - prefer search_memory for specific queries.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (default: 'default')",
                            "default": "default"
                        }
                    }
                }
            ),
            Tool(
                name="delete_memory",
                description="Delete a specific memory by ID. Use when user asks to forget something.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to delete"
                        }
                    },
                    "required": ["memory_id"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            mem = get_memory()
            
            if name == "add_memory":
                content = arguments["content"]
                user_id = arguments.get("user_id", "default")
                
                logger.info(f"Adding memory for user {user_id}: {content[:50]}...")
                result = mem.add(content, user_id=user_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": "Memory stored successfully",
                        "result": result
                    }, indent=2)
                )]

            elif name == "search_memory":
                query = arguments["query"]
                user_id = arguments.get("user_id", "default")
                limit = arguments.get("limit", 10)
                
                logger.info(f"Searching memories for user {user_id}: {query}")
                results = mem.search(query, user_id=user_id, limit=limit)
                
                # Format results for readability
                formatted = []
                for r in results:
                    formatted.append({
                        "id": r.get("id"),
                        "memory": r.get("memory"),
                        "score": r.get("score"),
                        "created_at": r.get("created_at")
                    })
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "query": query,
                        "total": len(formatted),
                        "memories": formatted
                    }, indent=2)
                )]

            elif name == "get_all_memories":
                user_id = arguments.get("user_id", "default")
                
                logger.info(f"Getting all memories for user {user_id}")
                results = mem.get_all(user_id=user_id)
                
                formatted = []
                for r in results:
                    formatted.append({
                        "id": r.get("id"),
                        "memory": r.get("memory"),
                        "created_at": r.get("created_at")
                    })
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "user_id": user_id,
                        "total": len(formatted),
                        "memories": formatted
                    }, indent=2)
                )]

            elif name == "delete_memory":
                memory_id = arguments["memory_id"]
                
                logger.info(f"Deleting memory: {memory_id}")
                mem.delete(memory_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": f"Memory {memory_id} deleted"
                    }, indent=2)
                )]

            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"})
                )]

        except Exception as e:
            logger.error(f"Error in tool {name}: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "tool": name
                }, indent=2)
            )]

    return server


# FastAPI app
app = FastAPI(title="MCP Memory Server")


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP communication."""
    logger.info("SSE connection established")
    
    # Create transport and server
    transport = SseServerTransport("/messages")
    server = create_mcp_server()
    
    # Store session
    session_id = id(transport)
    sessions[session_id] = {"transport": transport, "server": server}
    
    async def event_generator():
        try:
            # Connect server to transport
            async with transport.connect_sse(
                request.scope,
                request.receive,
                request._send
            ) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"SSE error: {e}")
        finally:
            logger.info("SSE connection closed")
            sessions.pop(session_id, None)
    
    return EventSourceResponse(event_generator())


@app.post("/messages")
async def messages_endpoint(request: Request):
    """Handle incoming MCP messages."""
    # Find the session from query params
    session_id = request.query_params.get("sessionId")
    
    if not session_id or int(session_id) not in sessions:
        return Response(
            content=json.dumps({"error": "Invalid or missing sessionId"}),
            status_code=400,
            media_type="application/json"
        )
    
    session = sessions[int(session_id)]
    # Handle the message through transport
    body = await request.body()
    # Process through the transport
    return Response(content="OK", status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "mcp-memory"}


if __name__ == "__main__":
    logger.info(f"Starting MCP Memory Server on port {PORT}")
    logger.info(f"Ollama host: {OLLAMA_HOST}")
    logger.info(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"LLM model: {LLM_MODEL}, Embed model: {EMBED_MODEL}")
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
