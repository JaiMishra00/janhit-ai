"""
Configuration for LLM, embeddings, and vector store.
"""

import os

# ========== LM STUDIO ========== 

# Comment this if running using APIs
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_MODEL = "meta-llama-3.1-8b-instruct"
EMBEDDING_MODEL = "text-embedding-bge-m3"

# LLM API (Hosted)
# LLM_BASE_URL = "https://api.openai.com/v1"
# LLM_MODEL = "gpt-4o-mini"
# LLM_API_KEY = os.getenv("OPENAI_API_KEY")

# ========== QDRANT ==========
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "md_bge_m3_source"
MEMORY_COLLECTION = "conversation_memory"

# ========== TESSERACT ==========
# Set this path if Tesseract is not in system PATH
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Windows default
)

POPPLER_CMD = os.getenv(
    "POPPLER_CMD"
    r"C:\\Users\\HP\\poppler\\Library\\bin"
)

# ========== CHUNKING ==========
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# ========== RETRIEVAL ==========
DEFAULT_TOP_K = 5
