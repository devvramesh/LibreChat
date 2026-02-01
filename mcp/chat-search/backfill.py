#!/usr/bin/env python3
"""
Backfill Script - Index existing Meilisearch messages into Qdrant for vector search.

Run this once after setting up the hybrid search to index existing messages:
    docker exec mcp_chat_search python backfill.py
"""

import os
import sys
import hashlib
import asyncio
import logging
from typing import List

import httpx
import meilisearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# =============================================================================
# CONFIGURATION
# =============================================================================

MEILI_URL = os.environ.get("MEILI_URL", "http://chat-meilisearch:7700")
MEILI_KEY = os.environ.get("MEILI_MASTER_KEY", "")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")

COLLECTION_NAME = "chat_messages"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
BATCH_SIZE = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CLIENTS
# =============================================================================

meili_client = meilisearch.Client(MEILI_URL, MEILI_KEY)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection():
    """Create Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' exists")
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}'")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )


async def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text[:8000]},
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []


def message_id_to_int(message_id: str) -> int:
    """Convert message ID string to integer for Qdrant."""
    # Use MD5 hash truncated to 63 bits (Qdrant uses int64)
    hash_bytes = hashlib.md5(message_id.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


async def index_message(message: dict) -> bool:
    """Index a single message to Qdrant."""
    text = message.get("text", "")
    message_id = message.get("messageId")
    
    if not text or not message_id:
        return False
    
    # Get embedding
    embedding = await get_embedding(text)
    if not embedding:
        return False
    
    # Upsert to Qdrant
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=message_id_to_int(message_id),
                    vector=embedding,
                    payload={
                        "messageId": message_id,
                        "conversationId": message.get("conversationId"),
                        "userId": message.get("user", "default"),
                        "sender": message.get("sender"),
                        "text": text[:2000],  # Truncate for storage
                        "createdAt": message.get("createdAt"),
                    }
                )
            ]
        )
        return True
    except Exception as e:
        logger.error(f"Qdrant upsert error: {e}")
        return False


async def backfill():
    """Backfill all messages from Meilisearch to Qdrant."""
    logger.info("=" * 60)
    logger.info("Starting backfill: Meilisearch → Qdrant")
    logger.info("=" * 60)
    
    # Ensure collection exists
    ensure_collection()
    
    # Get message index
    try:
        index = meili_client.index("messages")
        stats = index.get_stats()
        total_docs = stats.number_of_documents
        logger.info(f"Found {total_docs} messages in Meilisearch")
    except Exception as e:
        logger.error(f"Failed to connect to Meilisearch: {e}")
        return
    
    if total_docs == 0:
        logger.info("No messages to index")
        return
    
    # Process in batches
    offset = 0
    indexed = 0
    failed = 0
    
    while offset < total_docs:
        try:
            # Get batch of documents
            results = index.get_documents({
                "limit": BATCH_SIZE,
                "offset": offset,
                "fields": ["messageId", "conversationId", "sender", "text", "createdAt", "user"]
            })
            
            docs = results.results if hasattr(results, 'results') else []
            
            if not docs:
                break
            
            # Index each document
            for doc in docs:
                # Convert Meilisearch document to dict
                doc_dict = doc if isinstance(doc, dict) else doc.__dict__
                
                success = await index_message(doc_dict)
                if success:
                    indexed += 1
                else:
                    failed += 1
                
                # Progress update
                if (indexed + failed) % 100 == 0:
                    logger.info(f"Progress: {indexed + failed}/{total_docs} (indexed: {indexed}, failed: {failed})")
            
            offset += BATCH_SIZE
            
            # Small delay to avoid overwhelming Ollama
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Batch error at offset {offset}: {e}")
            offset += BATCH_SIZE
    
    logger.info("=" * 60)
    logger.info(f"Backfill complete!")
    logger.info(f"  Indexed: {indexed}")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"  Total:   {indexed + failed}")
    logger.info("=" * 60)
    
    # Verify collection
    try:
        collection = qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"Qdrant collection now has {collection.points_count} vectors")
    except Exception as e:
        logger.error(f"Failed to verify collection: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Chat Search Backfill - Meilisearch → Qdrant")
    print("=" * 60)
    print(f"Meilisearch: {MEILI_URL}")
    print(f"Qdrant:      {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Ollama:      {OLLAMA_HOST}")
    print(f"Model:       {EMBEDDING_MODEL}")
    print("=" * 60)
    
    asyncio.run(backfill())
