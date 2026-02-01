#!/usr/bin/env python3
"""
MCP Memory Server using Mem0 for intelligent memory extraction and semantic retrieval.

LLM: Configurable - Anthropic (fast), Bedrock (future), or Ollama (slow fallback)
Embeddings: Ollama nomic-embed-text (fast, free)
Vector Store: Qdrant
"""

import os
import json
import asyncio
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse
import uvicorn

from mem0 import Memory

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# LLM Provider: "anthropic", "bedrock", "ollama"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")

# Anthropic config
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

# Bedrock config (for future use)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")

# Ollama config
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Qdrant config
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# nomic-embed-text produces 768-dimensional embeddings
EMBED_DIMS = 768

PORT = int(os.environ.get("PORT", "3003"))

# Thread pool for sync Mem0 operations
executor = ThreadPoolExecutor(max_workers=4)

# =============================================================================
# MEM0 CONFIG BUILDER
# =============================================================================

def get_llm_config() -> dict:
    """Get LLM config based on provider setting."""
    
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY required when LLM_PROVIDER=anthropic")
        
        logger.info(f"[Memory] Using Anthropic LLM: {ANTHROPIC_MODEL}")
        return {
            "provider": "anthropic",
            "config": {
                "model": ANTHROPIC_MODEL,
                "api_key": ANTHROPIC_API_KEY,
                "temperature": 0.1,
                "max_tokens": 1500
            }
        }
    
    elif LLM_PROVIDER == "bedrock":
        logger.info(f"[Memory] Using AWS Bedrock LLM: {BEDROCK_MODEL}")
        return {
            "provider": "aws_bedrock",
            "config": {
                "model": BEDROCK_MODEL,
                "region": AWS_REGION,
                "temperature": 0.1,
                "max_tokens": 1500
            }
        }
    
    elif LLM_PROVIDER == "ollama":
        logger.info(f"[Memory] Using Ollama LLM: {OLLAMA_LLM_MODEL} (WARNING: may be slow)")
        return {
            "provider": "ollama",
            "config": {
                "model": OLLAMA_LLM_MODEL,
                "ollama_base_url": OLLAMA_HOST,
                "temperature": 0.1
            }
        }
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'anthropic', 'bedrock', or 'ollama'")


def build_mem0_config() -> dict:
    """
    Build Mem0 configuration based on LLM_PROVIDER env var.
    
    Embeddings always use Ollama (fast, free).
    LLM can be anthropic, bedrock, or ollama.
    """
    
    config = {
        # Vector store - always Qdrant
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": QDRANT_HOST,
                "port": QDRANT_PORT,
                "embedding_model_dims": EMBED_DIMS,
            }
        },
        
        # Embeddings - always Ollama (fast enough, free)
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": OLLAMA_EMBED_MODEL,
                "ollama_base_url": OLLAMA_HOST,
                "embedding_dims": EMBED_DIMS,
            }
        },
        
        # LLM - configurable
        "llm": get_llm_config()
    }
    
    return config


# =============================================================================
# INITIALIZE MEM0 (lazy)
# =============================================================================

memory: Optional[Memory] = None


def get_memory() -> Memory:
    """Get or initialize the Mem0 client."""
    global memory
    if memory is None:
        logger.info(f"[Memory] Initializing with LLM_PROVIDER={LLM_PROVIDER}")
        config = build_mem0_config()
        memory = Memory.from_config(config)
        logger.info("[Memory] Mem0 initialized successfully")
    return memory


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "name": "add_memory",
        "description": "Store important information about the user (preferences, facts, context, projects). Use this to remember things for future conversations. Do NOT store trivial or temporary information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember (e.g., 'User prefers dark mode', 'User is working on a LibreChat project')"
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
        "description": "Search stored memories semantically. Use this to recall user preferences, past context, or stored facts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g., 'programming preferences', 'current projects')"
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default')",
                    "default": "default"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_all_memories",
        "description": "Retrieve all stored memories for a user. Use sparingly - prefer search_memory for specific queries.",
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
        "description": "Delete a specific memory by its ID. Use when user asks to forget something.",
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


# =============================================================================
# TOOL EXECUTION
# =============================================================================

def call_tool_sync(name: str, arguments: dict) -> dict:
    """Execute a tool synchronously (runs in thread pool)."""
    try:
        mem = get_memory()

        if name == "add_memory":
            content = arguments.get("content", "")
            user_id = arguments.get("user_id", "default")

            if not content:
                return {"content": [{"type": "text", "text": "Error: content is required"}], "isError": True}

            logger.info(f"[Memory] Adding memory for user {user_id}: {content[:50]}...")
            start_time = datetime.now()
            
            result = mem.add(content, user_id=user_id)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"[Memory] Memory added in {elapsed:.2f}s")

            return {
                "content": [{
                    "type": "text",
                    "text": f"Memory stored successfully in {elapsed:.2f}s. Result: {json.dumps(result, default=str)}"
                }]
            }

        elif name == "search_memory":
            query = arguments.get("query", "")
            user_id = arguments.get("user_id", "default")
            limit = arguments.get("limit", 5)

            if not query:
                return {"content": [{"type": "text", "text": "Error: query is required"}], "isError": True}

            logger.info(f"[Memory] Searching memories for user {user_id}: {query}")
            response = mem.search(query, user_id=user_id, limit=limit)
            
            # Mem0 returns {'results': [...]} dict
            results = response.get("results", []) if isinstance(response, dict) else response

            if not results:
                return {"content": [{"type": "text", "text": f"No memories found for query: {query}"}]}

            # Format results
            formatted = []
            for i, r in enumerate(results):
                if isinstance(r, dict):
                    score = r.get("score", "N/A")
                    text = r.get("memory", r.get("text", ""))
                    mem_id = r.get("id", "unknown")
                    formatted.append(f"[{i+1}] (score: {score:.3f if isinstance(score, float) else score}, id: {mem_id})\n{text}")

            return {
                "content": [{
                    "type": "text",
                    "text": f"Found {len(results)} memories:\n\n" + "\n\n---\n\n".join(formatted)
                }]
            }

        elif name == "get_all_memories":
            user_id = arguments.get("user_id", "default")

            logger.info(f"[Memory] Getting all memories for user {user_id}")
            response = mem.get_all(user_id=user_id)
            
            # Mem0 may return {'results': [...]} dict or list
            results = response.get("results", []) if isinstance(response, dict) else response

            if not results:
                return {"content": [{"type": "text", "text": f"No memories stored for user: {user_id}"}]}

            formatted = []
            for i, r in enumerate(results):
                if isinstance(r, dict):
                    text = r.get("memory", r.get("text", ""))
                    mem_id = r.get("id", "unknown")
                    created = r.get("created_at", "unknown")
                    formatted.append(f"[{i+1}] (id: {mem_id}, created: {created})\n{text}")

            return {
                "content": [{
                    "type": "text",
                    "text": f"All memories ({len(results)} total):\n\n" + "\n\n---\n\n".join(formatted)
                }]
            }

        elif name == "delete_memory":
            memory_id = arguments.get("memory_id", "")

            if not memory_id:
                return {"content": [{"type": "text", "text": "Error: memory_id is required"}], "isError": True}

            logger.info(f"[Memory] Deleting memory: {memory_id}")
            mem.delete(memory_id)

            return {
                "content": [{
                    "type": "text",
                    "text": f"Memory {memory_id} deleted successfully"
                }]
            }

        else:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}

    except Exception as e:
        logger.error(f"[Memory] Error in tool {name}: {e}", exc_info=True)
        return {
            "content": [{"type": "text", "text": f"Error: {str(e)}"}],
            "isError": True
        }


async def call_tool_async(name: str, arguments: dict) -> dict:
    """Execute a tool asynchronously using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, call_tool_sync, name, arguments)


# =============================================================================
# MCP MESSAGE HANDLING
# =============================================================================

# MCP Server info
SERVER_INFO = {
    "name": "memory",
    "version": "1.0.0",
    "protocolVersion": "2024-11-05"
}


def handle_message_sync(message: dict, session_id: str) -> Optional[dict]:
    """Handle an incoming MCP message and return the response."""
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params", {})

    logger.info(f"[{session_id}] Handling method: {method}, id: {msg_id}")

    if method == "initialize":
        return {
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

    elif method == "notifications/initialized":
        logger.info(f"[{session_id}] Received initialized notification")
        return None

    elif method == "tools/list":
        logger.info(f"[{session_id}] Sending tools list with {len(TOOLS)} tools")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"[{session_id}] Calling tool: {tool_name}")
        result = call_tool_sync(tool_name, arguments)

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


async def handle_message_async(message: dict, session_id: str) -> Optional[dict]:
    """Handle message asynchronously - tools run in thread pool."""
    method = message.get("method")
    
    # Tool calls run in thread pool for async behavior
    if method == "tools/call":
        msg_id = message.get("id")
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"[{session_id}] Calling tool async: {tool_name}")
        result = await call_tool_async(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result
        }
    
    # Other methods are fast, run sync
    return handle_message_sync(message, session_id)


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="MCP Memory Server")

# Store active sessions
sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP communication."""
    session_id = str(uuid.uuid4())
    logger.info(f"[Memory] SSE connection request: {session_id}")

    event_queue: asyncio.Queue = asyncio.Queue()
    sessions[session_id] = {
        "connected": True,
        "queue": event_queue
    }

    async def event_generator():
        try:
            # Send the endpoint event
            endpoint_url = f"/messages?sessionId={session_id}"
            yield f"event: endpoint\ndata: {endpoint_url}\n\n"

            # Keep connection alive and send queued messages
            while sessions.get(session_id, {}).get("connected", False):
                try:
                    msg = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                    msg_json = json.dumps(msg)
                    logger.info(f"[{session_id}] Sending SSE message: {msg_json[:200]}")
                    yield f"event: message\ndata: {msg_json}\n\n"
                except asyncio.TimeoutError:
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
        }
    )


@app.post("/messages")
async def messages_endpoint(request: Request):
    """Handle incoming MCP messages. Responses sent via SSE."""
    session_id = request.query_params.get("sessionId")

    if not session_id or session_id not in sessions:
        return Response(
            content=json.dumps({"error": "Invalid or missing sessionId"}),
            status_code=400,
            media_type="application/json"
        )

    try:
        body = await request.json()
        logger.info(f"[{session_id}] Message: {body.get('method', 'unknown')}")

        response = await handle_message_async(body, session_id)

        if response is not None:
            session = sessions.get(session_id)
            if session and session.get("queue"):
                await session["queue"].put(response)

        return Response(status_code=202)

    except Exception as e:
        logger.error(f"[{session_id}] Error handling message: {e}", exc_info=True)
        error_response = {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": str(e)}
        }
        session = sessions.get(session_id)
        if session and session.get("queue"):
            await session["queue"].put(error_response)
        return Response(status_code=202)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "mcp-memory",
        "llm_provider": LLM_PROVIDER,
        "llm_model": ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else BEDROCK_MODEL if LLM_PROVIDER == "bedrock" else OLLAMA_LLM_MODEL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "sessions": len(sessions)
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info(f"[Memory] Starting MCP Memory Server on port {PORT}")
    logger.info(f"[Memory] LLM Provider: {LLM_PROVIDER}")
    if LLM_PROVIDER == "anthropic":
        logger.info(f"[Memory] Anthropic Model: {ANTHROPIC_MODEL}")
    elif LLM_PROVIDER == "bedrock":
        logger.info(f"[Memory] Bedrock Model: {BEDROCK_MODEL}")
    else:
        logger.info(f"[Memory] Ollama LLM Model: {OLLAMA_LLM_MODEL}")
    logger.info(f"[Memory] Ollama Embed Model: {OLLAMA_EMBED_MODEL}")
    logger.info(f"[Memory] Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
