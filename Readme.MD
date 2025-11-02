# InsightPDF 📄🤖

A PDF Question Answering system built using **LangChain**, **Astra DB Vector Search (REST API)** and **MiniLM embeddings**, enabling semantic search + context aware responses through **Retrieval Augmented Generation (RAG)**.

This project converts PDFs into vector embeddings, stores them inside Astra DB Vector Database, and retrieves relevant text chunks dynamically to answer user queries with GPT-4-mini.

---

## ⚙️ Tech Stack

| Layer | Tech |
|-------|------|
| Orchestration | LangChain |
| Vector DB | Astra DB (Vector Search - REST) |
| Embeddings | MiniLM (`all-MiniLM-L6-v2`) |
| LLM | GPT-4-mini (OpenAI API) |
| PDF Loader | PyPDF |
| Similarity Search | Astra Vector |

---

## Features

- PDF → Text → Chunking → Embeddings pipeline
- Astra DB Vector Store storage
- Fast semantic similarity search
- GPT-4 mini powered answer generation
- Modular + Resume friendly RAG code base
- Can be extended to Legal / Finance / Enterprise Document Search

---

## Project Structure

