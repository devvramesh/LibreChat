#!/usr/bin/env python3
"""
MCP Chat Search Server - Hybrid Search (Keyword + Semantic + Temporal)

Architecture:
Query
  ├── [1] Meilisearch BM25 Search → Top 30 keyword results
  ├── [2] Qdrant Vector Search → Top 30 semantic results  
  └── [3] RRF Fusion → Merged & ranked results
              ↓
        [4] Temporal Boost → Recency-adjusted scores
              ↓
        Return Top N
"""

import os
import json
import asyncio
import logging
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse
import uvicorn
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
import meilisearch

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MEILI_URL = os.environ.get("MEILI_URL", "http://chat-meilisearch:7700")
MEILI_KEY = os.environ.get("MEILI_MASTER_KEY", "")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
PORT = int(os.environ.get("PORT", "3001"))

COLLECTION_NAME = "chat_messages"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # nomic-embed-text dimension

# Thread pool for sync operations
executor = ThreadPoolExecutor(max_workers=4)

# =============================================================================
# CLIENTS
# =============================================================================

meili_client: Optional[meilisearch.Client] = None
qdrant_client: Optional[QdrantClient] = None


def get_meili_client() -> meilisearch.Client:
    global meili_client
    if meili_client is None:
        meili_client = meilisearch.Client(MEILI_URL, MEILI_KEY)
        logger.info(f"[ChatSearch] Connected to Meilisearch at {MEILI_URL}")
    return meili_client


def get_qdrant_client() -> QdrantClient:
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"[ChatSearch] Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        # Ensure collection exists
        try:
            qdrant_client.get_collection(COLLECTION_NAME)
        except Exception:
            logger.info(f"[ChatSearch] Creating collection {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
    return qdrant_client


# =============================================================================
# EMBEDDING
# =============================================================================

async def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text[:8000]},  # Truncate long text
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"[ChatSearch] Embedding error: {e}")
            return []


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def keyword_search_sync(
    query: str,
    limit: int = 30,
    user_filter: Optional[str] = None,
    conversation_id: Optional[str] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None
) -> List[Dict]:
    """Keyword search via Meilisearch BM25."""
    try:
        client = get_meili_client()
        index = client.index("messages")
        
        # Build filter
        filters = []
        if user_filter:
            filters.append(f'user = "{user_filter}"')
        if conversation_id:
            filters.append(f'conversationId = "{conversation_id}"')
        # Note: Meilisearch time filtering requires filterable attributes to be set
        
        search_params = {
            "limit": limit,
            "attributesToRetrieve": ["messageId", "conversationId", "sender", "text", "createdAt", "user"]
        }
        
        if filters:
            search_params["filter"] = " AND ".join(filters)
        
        results = index.search(query, search_params)
        return results.get("hits", [])
    except Exception as e:
        logger.error(f"[ChatSearch] Meilisearch error: {e}")
        return []


async def keyword_search(
    query: str,
    limit: int = 30,
    user_filter: Optional[str] = None,
    conversation_id: Optional[str] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None
) -> List[Dict]:
    """Async wrapper for keyword search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        keyword_search_sync,
        query, limit, user_filter, conversation_id, after, before
    )


def vector_search_sync(
    query_embedding: List[float],
    limit: int = 30,
    user_filter: Optional[str] = None,
    conversation_id: Optional[str] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None
) -> List[Dict]:
    """Vector search via Qdrant."""
    if not query_embedding:
        return []
    
    try:
        client = get_qdrant_client()
        
        # Build filter conditions
        conditions = []
        if user_filter:
            conditions.append(FieldCondition(key="userId", match=MatchValue(value=user_filter)))
        if conversation_id:
            conditions.append(FieldCondition(key="conversationId", match=MatchValue(value=conversation_id)))
        
        query_filter = Filter(must=conditions) if conditions else None
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True
        )
        
        # Convert to dict format matching Meilisearch output
        return [
            {
                "messageId": hit.payload.get("messageId"),
                "conversationId": hit.payload.get("conversationId"),
                "sender": hit.payload.get("sender"),
                "text": hit.payload.get("text"),
                "createdAt": hit.payload.get("createdAt"),
                "user": hit.payload.get("userId"),
                "_vector_score": hit.score
            }
            for hit in results
        ]
    except Exception as e:
        logger.error(f"[ChatSearch] Qdrant error: {e}")
        return []


async def vector_search(
    query_embedding: List[float],
    limit: int = 30,
    user_filter: Optional[str] = None,
    conversation_id: Optional[str] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None
) -> List[Dict]:
    """Async wrapper for vector search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        vector_search_sync,
        query_embedding, limit, user_filter, conversation_id, after, before
    )


# =============================================================================
# RRF FUSION
# =============================================================================

def rrf_fusion(keyword_results: List[Dict], vector_results: List[Dict], k: int = 60) -> List[Dict]:
    """
    Reciprocal Rank Fusion to combine keyword and vector search results.
    RRF score = sum(1 / (k + rank)) for each result list
    """
    scores: Dict[str, Dict] = {}
    
    # Score keyword results
    for rank, result in enumerate(keyword_results):
        doc_id = result.get("messageId")
        if not doc_id:
            continue
        if doc_id not in scores:
            scores[doc_id] = {"data": result, "score": 0}
        scores[doc_id]["score"] += 1 / (k + rank + 1)
    
    # Score vector results
    for rank, result in enumerate(vector_results):
        doc_id = result.get("messageId")
        if not doc_id:
            continue
        if doc_id not in scores:
            scores[doc_id] = {"data": result, "score": 0}
        scores[doc_id]["score"] += 1 / (k + rank + 1)
    
    # Sort by combined score
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["data"] for item in sorted_results]


# =============================================================================
# TEMPORAL BOOST
# =============================================================================

def apply_temporal_boost(results: List[Dict], recency_weight: float = 0.2) -> List[Dict]:
    """
    Boost recent results. Results with timestamps get a recency boost.
    recency_weight: 0-1, how much to weight recency vs relevance
    """
    now = datetime.now(timezone.utc)
    max_age_days = 365  # Normalize against 1 year
    
    for i, result in enumerate(results):
        # Calculate position-based score (higher for earlier positions)
        position_score = 1 / (i + 1)
        
        # Calculate recency score
        recency_score = 0
        created_at = result.get("createdAt")
        if created_at:
            try:
                if isinstance(created_at, str):
                    # Parse ISO format
                    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    created = created_at
                
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                
                age_days = (now - created).days
                recency_score = max(0, 1 - (age_days / max_age_days))
            except Exception:
                pass
        
        result["_final_score"] = position_score + (recency_score * recency_weight)
    
    return sorted(results, key=lambda x: x.get("_final_score", 0), reverse=True)


# =============================================================================
# TIME FILTER HELPERS
# =============================================================================

def parse_time_range(time_range: Optional[str]) -> tuple[Optional[datetime], Optional[datetime]]:
    """Convert time_range string to after/before datetimes."""
    if not time_range:
        return None, None
    
    now = datetime.now(timezone.utc)
    
    if time_range == "day":
        after = now - timedelta(days=1)
    elif time_range == "week":
        after = now - timedelta(weeks=1)
    elif time_range == "month":
        after = now - timedelta(days=30)
    elif time_range == "year":
        after = now - timedelta(days=365)
    else:
        return None, None
    
    return after, None


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "name": "search_chat_history",
        "description": "Search past conversations using hybrid search (keyword + semantic). Combines BM25 keyword matching with vector similarity for best results. Use for 'what did we discuss about X' or 'find our conversation about Y'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in chat history"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10, max: 20)",
                    "default": 10
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Filter by time period - 'day', 'week', 'month', 'year'"
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Filter to specific conversation ID"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "recent_chats",
        "description": "Get recent chat messages, sorted by time (newest first). Use for time-based retrieval like 'what did we discuss yesterday' or 'show recent conversations'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of recent messages to return (default: 5, max: 20)",
                    "default": 5
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month"],
                    "description": "Filter by time period"
                }
            }
        }
    }
]


# =============================================================================
# TOOL EXECUTION
# =============================================================================

async def call_tool(name: str, arguments: dict) -> dict:
    """Execute a tool and return the result."""
    try:
        if name == "search_chat_history":
            query = arguments.get("query", "")
            limit = min(arguments.get("limit", 10), 20)
            time_range = arguments.get("time_range")
            conversation_id = arguments.get("conversation_id")
            
            if not query:
                return {"content": [{"type": "text", "text": "Error: query is required"}], "isError": True}
            
            logger.info(f"[ChatSearch] Hybrid search: {query}")
            
            # Parse time filter
            after, before = parse_time_range(time_range)
            
            # 1. Keyword search via Meilisearch
            keyword_results = await keyword_search(
                query, limit=30, conversation_id=conversation_id, after=after, before=before
            )
            logger.info(f"[ChatSearch] Keyword results: {len(keyword_results)}")
            
            # 2. Vector search via Qdrant
            query_embedding = await get_embedding(query)
            vector_results = []
            if query_embedding:
                vector_results = await vector_search(
                    query_embedding, limit=30, conversation_id=conversation_id, after=after, before=before
                )
                logger.info(f"[ChatSearch] Vector results: {len(vector_results)}")
            
            # 3. RRF Fusion
            if keyword_results and vector_results:
                fused_results = rrf_fusion(keyword_results, vector_results)
                logger.info(f"[ChatSearch] Fused results: {len(fused_results)}")
            elif keyword_results:
                fused_results = keyword_results
            elif vector_results:
                fused_results = vector_results
            else:
                return {"content": [{"type": "text", "text": f"No results found for: {query}"}]}
            
            # 4. Apply temporal boost
            boosted_results = apply_temporal_boost(fused_results)
            
            # 5. Return top N
            final_results = boosted_results[:limit]
            
            # Format response
            formatted = []
            for i, r in enumerate(final_results):
                text = r.get("text", "")[:500]  # Truncate long messages
                created = r.get("createdAt", "Unknown date")
                sender = r.get("sender", "Unknown")
                conv_id = r.get("conversationId", "")[:8] if r.get("conversationId") else ""
                formatted.append(f"[{i+1}] {created} ({sender}) [conv:{conv_id}...]\n{text}")
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"Found {len(final_results)} results for '{query}':\n\n" + "\n\n---\n\n".join(formatted)
                }]
            }
        
        elif name == "recent_chats":
            n = min(arguments.get("n", 5), 20)
            time_range = arguments.get("time_range")
            
            after, before = parse_time_range(time_range)
            
            logger.info(f"[ChatSearch] Recent chats: {n}")
            
            # Use Meilisearch with empty query, sorted by date
            try:
                client = get_meili_client()
                index = client.index("messages")
                
                results = index.search("", {
                    "limit": n,
                    "sort": ["createdAt:desc"],
                    "attributesToRetrieve": ["messageId", "conversationId", "sender", "text", "createdAt"]
                })
                
                hits = results.get("hits", [])
                
                if not hits:
                    return {"content": [{"type": "text", "text": "No recent messages found."}]}
                
                formatted = []
                for i, r in enumerate(hits):
                    text = r.get("text", "")[:500]
                    created = r.get("createdAt", "Unknown date")
                    sender = r.get("sender", "Unknown")
                    formatted.append(f"[{i+1}] {created} ({sender})\n{text}")
                
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Recent {len(hits)} messages:\n\n" + "\n\n---\n\n".join(formatted)
                    }]
                }
            except Exception as e:
                logger.error(f"[ChatSearch] Recent chats error: {e}")
                return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}
        
        else:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
    
    except Exception as e:
        logger.error(f"[ChatSearch] Tool error: {e}", exc_info=True)
        return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}


# =============================================================================
# MCP MESSAGE HANDLING
# =============================================================================

SERVER_INFO = {
    "name": "chat-search",
    "version": "2.0.0",
    "protocolVersion": "2024-11-05"
}


async def handle_message(message: dict, session_id: str) -> Optional[dict]:
    """Handle an incoming MCP message."""
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params", {})
    
    logger.info(f"[{session_id}] Method: {method}")
    
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
        return None
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS}
        }
    
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        result = await call_tool(tool_name, arguments)
        
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
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="MCP Chat Search Server")

sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP communication."""
    session_id = str(uuid.uuid4())
    logger.info(f"[ChatSearch] SSE connection: {session_id}")
    
    event_queue: asyncio.Queue = asyncio.Queue()
    sessions[session_id] = {"connected": True, "queue": event_queue}
    
    async def event_generator():
        try:
            yield f"event: endpoint\ndata: /messages?sessionId={session_id}\n\n"
            
            while sessions.get(session_id, {}).get("connected", False):
                try:
                    msg = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            sessions.pop(session_id, None)
            logger.info(f"[{session_id}] SSE closed")
    
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
    """Handle incoming MCP messages."""
    session_id = request.query_params.get("sessionId")
    
    if not session_id or session_id not in sessions:
        return Response(
            content=json.dumps({"error": "Invalid sessionId"}),
            status_code=400,
            media_type="application/json"
        )
    
    try:
        body = await request.json()
        response = await handle_message(body, session_id)
        
        if response is not None:
            session = sessions.get(session_id)
            if session and session.get("queue"):
                await session["queue"].put(response)
        
        return Response(status_code=202)
    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}")
        return Response(status_code=202)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "mcp-chat-search",
        "version": SERVER_INFO["version"],
        "meili_url": MEILI_URL,
        "qdrant": f"{QDRANT_HOST}:{QDRANT_PORT}",
        "ollama": OLLAMA_HOST,
        "sessions": len(sessions)
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info(f"[ChatSearch] Starting on port {PORT}")
    logger.info(f"[ChatSearch] Meilisearch: {MEILI_URL}")
    logger.info(f"[ChatSearch] Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"[ChatSearch] Ollama: {OLLAMA_HOST}")
    
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
