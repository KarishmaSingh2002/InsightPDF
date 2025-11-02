# src/astra_adapter.py
import os
import requests
import uuid
import numpy as np
from src.config import ASTRA_BASE_URL, ASTRA_DB_TOKEN, ASTRA_KEYSPACE, ASTRA_TABLE

HEADERS = {
    "Content-Type": "application/json",
}
if ASTRA_DB_TOKEN:
    HEADERS["X-Cassandra-Token"] = ASTRA_DB_TOKEN

class AstraVectorStore:
    """
    Minimal REST adapter for Astra Vector Search.
    IMPORTANT: Replace vector_search_endpoint with the exact endpoint from Astra console.
    """
    def __init__(self):
        self.enabled = bool(ASTRA_BASE_URL and ASTRA_DB_TOKEN)
        if not self.enabled:
            return
        self.base_url = ASTRA_BASE_URL.rstrip("/")
        self.keyspace = ASTRA_KEYSPACE
        self.table = ASTRA_TABLE
        self.upsert_url = f"{self.base_url}/api/rest/v2/keyspaces/{self.keyspace}/tables/{self.table}"
        # Placeholder path for vector-search — change to your Astra vector-search endpoint
        self.vector_search_endpoint = f"{self.base_url}/api/rest/v2/keyspaces/{self.keyspace}/tables/{self.table}/vector-search"

    def upsert_vector(self, vector, metadata=None, content=None, id=None):
        if not self.enabled:
            raise RuntimeError("Astra adapter not configured. Set ASTRA_BASE_URL and ASTRA_DB_TOKEN in .env")
        payload = {
            "id": id or str(uuid.uuid4()),
            "content": content or "",
            "metadata": metadata or {},
            "embedding": [float(x) for x in (vector.tolist() if hasattr(vector, "tolist") else vector)]
        }
        r = requests.post(self.upsert_url, headers=HEADERS, json=payload, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Astra upsert failed: {r.status_code} {r.text}")
        return r.json()

    def query_vector(self, query_vector, top_k=5, threshold=None):
        if not self.enabled:
            raise RuntimeError("Astra adapter not configured. Set ASTRA_BASE_URL and ASTRA_DB_TOKEN in .env")
        payload = {
            "vector": [float(x) for x in (query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector)],
            "top_k": top_k
        }
        if threshold is not None:
            payload["threshold"] = float(threshold)
        r = requests.post(self.vector_search_endpoint, headers=HEADERS, json=payload, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Astra vector search failed: {r.status_code} {r.text}. Check endpoint.")
        return r.json()
