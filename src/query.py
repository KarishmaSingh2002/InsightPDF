# src/query.py
import os
import argparse
import pickle
import numpy as np
import faiss
from src.utils import load_embedding_model, embed_texts
from src.astra_adapter import AstraVectorStore
from src.config import PERSIST_DIR, OPENAI_CHAT_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_faiss(persist_dir=PERSIST_DIR):
    idx_path = os.path.join(persist_dir, "faiss.index")
    docs_path = os.path.join(persist_dir, "docs.pkl")
    if not os.path.exists(idx_path) or not os.path.exists(docs_path):
        raise FileNotFoundError("FAISS index or docs not found. Run ingest first.")
    index = faiss.read_index(idx_path)
    with open(docs_path, "rb") as f:
        data = pickle.load(f)
    return index, data["docs"], data["metas"]

def retrieve_faiss(query, index, docs, model, k=5):
    q_emb = embed_texts([query], model=model)
    D, I = index.search(q_emb, k)
    hits = []
    for idx in I[0]:
        if idx < 0 or idx >= len(docs):
            continue
        hits.append(docs[idx])
    return hits

def retrieve_astra(query, adapter, model, top_k=5):
    q_emb = embed_texts([query], model=model)[0]
    resp = adapter.query_vector(q_emb, top_k=top_k)
    hits = []
    for h in resp.get("hits", []):
        content = h.get("content") or h.get("document") or ""
        hits.append(content)
    return hits

def call_openai(system_prompt, user_prompt, temperature=OPENAI_TEMPERATURE, max_tokens=OPENAI_MAX_TOKENS):
    if openai.api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set.")
    resp = openai.ChatCompletion.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def interactive_loop(persist_dir=PERSIST_DIR, use_astra=False, k=5):
    model = load_embedding_model()
    adapter = AstraVectorStore() if use_astra else None
    if not use_astra:
        index, docs, metas = load_faiss(persist_dir)

    print("Type questions (type 'exit' to quit).")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if use_astra:
            hits = retrieve_astra(q, adapter, model, top_k=k)
        else:
            hits = retrieve_faiss(q, index, docs, model, k=k)
        context = "\n\n---\n\n".join(hits)
        system = "You are a helpful assistant. Use the provided context to answer the question. If not in context, say you don't know."
        user_msg = f"Context:\n{context}\n\nQuestion: {q}\nAnswer concisely and cite sources if possible."
        answer = call_openai(system, user_msg)
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n=== SOURCES ===\n")
        if not use_astra:
            sources = {m["source"] for m in metas}
            print(", ".join(sources))
        else:
            print("Sources from Astra (if provided in hits).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default=PERSIST_DIR)
    parser.add_argument("--use_astra", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    interactive_loop(persist_dir=args.persist_dir, use_astra=args.use_astra, k=args.k)
