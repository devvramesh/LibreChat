#!/usr/bin/env python3
"""
MCP Memory Server using Mem0 for intelligent memory extraction and semantic retrieval.
Uses Ollama for LLM and embeddings, Qdrant for vector storage.

This implementation manually implements the MCP SSE transport protocol.
IMPORTANT: Responses are sent via SSE events, not POST response bodies.
"""

import os
import json
import asyncio
import logging
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse
import uvicorn

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
# nomic-embed-text produces 768-dimensional embeddings
EMBED_DIMS = 768

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
            "embedding_dims": EMBED_DIMS,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "embedding_model_dims": EMBED_DIMS,
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


# Store active sessions with queues for SSE messages
sessions: Dict[str, Dict[str, Any]] = {}

# MCP Server info
SERVER_INFO = {
    "name": "memory",
    "version": "1.0.0",
    "protocolVersion": "2024-11-05"
}

# Tool definitions
TOOLS = [
    {
        "name": "add_memory",
        "description": "Store a new memory. Call this when the user shares important information about themselves, their preferences, projects, or anything worth remembering for future conversations. Do NOT store trivial or temporary information.",
        "inputSchema": {
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
    },
    {
        "name": "search_memory",
        "description": "Search memories semantically. Use this to recall information from past conversations. Returns relevant memories ranked by relevance.",
        "inputSchema": {
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
    },
    {
        "name": "get_all_memories",
        "description": "Get all stored memories for a user. Use sparingly - prefer search_memory for specific queries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default')",
                    "default": "default"
                }
            }
        }
    },
    {
        "name": "delete_memory",
        "description": "Delete a specific memory by ID. Use when user asks to forget something.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The ID of the memory to delete"
                }
            },
            "required": ["memory_id"]
        }
    }
]


def call_tool(name: str, arguments: dict) -> dict:
    """Execute a tool and return the result."""
    try:
        mem = get_memory()

        if name == "add_memory":
            content = arguments["content"]
            user_id = arguments.get("user_id", "default")

            logger.info(f"Adding memory for user {user_id}: {content[:50]}...")
            result = mem.add(content, user_id=user_id)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "status": "success",
                        "message": "Memory stored successfully",
                        "result": result
                    }, indent=2)
                }]
            }

        elif name == "search_memory":
            query = arguments["query"]
            user_id = arguments.get("user_id", "default")
            limit = arguments.get("limit", 10)

            logger.info(f"Searching memories for user {user_id}: {query}")
            response = mem.search(query, user_id=user_id, limit=limit)
            
            # Mem0 returns {'results': [...]} dict
            results = response.get("results", []) if isinstance(response, dict) else response

            formatted = []
            for r in results:
                if isinstance(r, dict):
                    formatted.append({
                        "id": r.get("id"),
                        "memory": r.get("memory"),
                        "score": r.get("score"),
                        "created_at": r.get("created_at")
                    })

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "query": query,
                        "total": len(formatted),
                        "memories": formatted
                    }, indent=2)
                }]
            }

        elif name == "get_all_memories":
            user_id = arguments.get("user_id", "default")

            logger.info(f"Getting all memories for user {user_id}")
            response = mem.get_all(user_id=user_id)
            
            # Mem0 may return {'results': [...]} dict or list
            results = response.get("results", []) if isinstance(response, dict) else response

            formatted = []
            for r in results:
                if isinstance(r, dict):
                    formatted.append({
                        "id": r.get("id"),
                        "memory": r.get("memory"),
                        "created_at": r.get("created_at")
                    })

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "user_id": user_id,
                        "total": len(formatted),
                        "memories": formatted
                    }, indent=2)
                }]
            }

        elif name == "delete_memory":
            memory_id = arguments["memory_id"]

            logger.info(f"Deleting memory: {memory_id}")
            mem.delete(memory_id)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "status": "success",
                        "message": f"Memory {memory_id} deleted"
                    }, indent=2)
                }]
            }

        else:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({"error": f"Unknown tool: {name}"})
                }],
                "isError": True
            }

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": str(e),
                    "tool": name
                }, indent=2)
            }],
            "isError": True
        }


def handle_message(message: dict, session_id: str) -> Optional[dict]:
    """Handle an incoming MCP message and return the response."""
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params", {})

    logger.info(f"[{session_id}] Handling method: {method}, id: {msg_id}")

    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": SERVER_INFO["protocolVersion"],
                "serverInfo": {
                    "name": SERVER_INFO["name"],
                    "version": SERVER_INFO["version"]
                },
                "capabilities": {
                    "tools": {"listChanged": True}
                }
            }
        }
        logger.info(f"[{session_id}] Sending initialize response")
        return response

    elif method == "notifications/initialized":
        # No response needed for notifications
        logger.info(f"[{session_id}] Received initialized notification")
        return None

    elif method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS
            }
        }
        logger.info(f"[{session_id}] Sending tools list with {len(TOOLS)} tools")
        return response

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"[{session_id}] Calling tool: {tool_name}")
        result = call_tool(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result
        }

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {}
        }

    else:
        logger.warning(f"[{session_id}] Unknown method: {method}")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }


# FastAPI app
app = FastAPI(title="MCP Memory Server")


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP communication."""
    session_id = str(uuid.uuid4())
    logger.info(f"SSE connection request: {session_id}")

    # Create a queue for this session's events
    event_queue: asyncio.Queue = asyncio.Queue()
    sessions[session_id] = {
        "connected": True,
        "queue": event_queue
    }

    async def event_generator():
        try:
            # Send the endpoint event immediately
            endpoint_url = f"/messages?sessionId={session_id}"
            endpoint_event = f"event: endpoint\ndata: {endpoint_url}\n\n"
            logger.info(f"[{session_id}] Sending endpoint event: {endpoint_url}")
            yield endpoint_event

            # Keep connection alive and send queued messages
            while sessions.get(session_id, {}).get("connected", False):
                try:
                    # Wait for messages with timeout
                    msg = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                    # Send message as SSE event
                    msg_json = json.dumps(msg)
                    logger.info(f"[{session_id}] Sending SSE message: {msg_json[:200]}")
                    yield f"event: message\ndata: {msg_json}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield ": keepalive\n\n"

        except asyncio.CancelledError:
            logger.info(f"[{session_id}] SSE generator cancelled")
        except Exception as e:
            logger.error(f"[{session_id}] SSE generator error: {e}")
        finally:
            sessions.pop(session_id, None)
            logger.info(f"[{session_id}] SSE connection closed")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/messages")
async def messages_endpoint(request: Request):
    """Handle incoming MCP messages.
    
    In MCP SSE transport, responses should be sent via the SSE stream,
    not as POST response body. This endpoint queues responses.
    """
    session_id = request.query_params.get("sessionId")
    logger.info(f"POST /messages received, sessionId: {session_id}")

    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid sessionId: {session_id}")
        return Response(
            content=json.dumps({"error": "Invalid or missing sessionId"}),
            status_code=400,
            media_type="application/json"
        )

    try:
        body = await request.json()
        logger.info(f"[{session_id}] Message body: {json.dumps(body)[:200]}")

        response = handle_message(body, session_id)

        if response is not None:
            # Queue the response to be sent via SSE
            session = sessions.get(session_id)
            if session and session.get("queue"):
                await session["queue"].put(response)
                logger.info(f"[{session_id}] Response queued for SSE delivery")

        # Return 202 Accepted - actual response goes via SSE
        return Response(status_code=202)

    except Exception as e:
        logger.error(f"[{session_id}] Error handling message: {e}", exc_info=True)
        # Queue error response via SSE
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }
        session = sessions.get(session_id)
        if session and session.get("queue"):
            await session["queue"].put(error_response)
        return Response(status_code=202)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "mcp-memory", "sessions": len(sessions)}


if __name__ == "__main__":
    logger.info(f"Starting MCP Memory Server on port {PORT}")
    logger.info(f"Ollama host: {OLLAMA_HOST}")
    logger.info(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"LLM model: {LLM_MODEL}, Embed model: {EMBED_MODEL}")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
