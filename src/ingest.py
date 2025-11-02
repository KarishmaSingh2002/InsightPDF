# src/ingest.py
import os
import argparse
from tqdm import tqdm
import pickle
import numpy as np
import faiss

from src.utils import load_embedding_model, embed_texts
from utils.pdf_loader import pdf_to_texts
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PERSIST_DIR
from src.astra_adapter import AstraVectorStore

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_faiss(index, docs, metas, persist_dir=PERSIST_DIR):
    os.makedirs(persist_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(persist_dir, "faiss.index"))
    with open(os.path.join(persist_dir, "docs.pkl"), "wb") as f:
        pickle.dump({"docs": docs, "metas": metas}, f)

def ingest_folder(pdf_dir, persist_dir=PERSIST_DIR, use_astra=False):
    model = load_embedding_model()
    all_texts = []
    metadatas = []

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        print(f"Reading {path}")
        text = pdf_to_texts(path)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            metadatas.append({"source": fname, "chunk": i})

    print(f"Total chunks: {len(all_texts)}")
    if len(all_texts) == 0:
        print("No chunks found. Exiting.")
        return

    embeddings = embed_texts(all_texts, model=model)  # already normalized
    if use_astra:
        adapter = AstraVectorStore()
        for emb, doc, meta in tqdm(zip(embeddings, all_texts, metadatas), total=len(all_texts)):
            adapter.upsert_vector(vector=emb, metadata=meta, content=doc)
        print("Uploaded vectors to Astra.")
    else:
        index = build_faiss_index(embeddings)
        save_faiss(index, all_texts, metadatas, persist_dir)
        print(f"Saved FAISS index + docs to {persist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True, help="Directory with PDFs")
    parser.add_argument("--persist_dir", default=PERSIST_DIR)
    parser.add_argument("--use_astra", action="store_true", help="Upload to Astra instead of FAISS")
    args = parser.parse_args()
    ingest_folder(args.pdf_dir, args.persist_dir, use_astra=args.use_astra)
