# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "600"))

# Astra REST
ASTRA_BASE_URL = os.getenv("ASTRA_BASE_URL")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_KEYSPACE = os.getenv("ASTRA_KEYSPACE", "insightpdf")
ASTRA_TABLE = os.getenv("ASTRA_TABLE", "vectors")

# Persistence
PERSIST_DIR = os.getenv("PERSIST_DIR", "data")

# Ingest defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))      # words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
