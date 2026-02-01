#!/bin/bash
# Initialize Ollama models for Mem0
# Run this after Ollama is up and running

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
LLM_MODEL="${LLM_MODEL:-llama3.1:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

echo "Waiting for Ollama to be ready..."
until curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    echo "Ollama not ready, waiting..."
    sleep 5
done
echo "Ollama is ready!"

echo "Pulling LLM model: ${LLM_MODEL}..."
curl -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"${LLM_MODEL}\"}"

echo "Pulling embedding model: ${EMBED_MODEL}..."
curl -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"${EMBED_MODEL}\"}"

echo "Models pulled successfully!"
