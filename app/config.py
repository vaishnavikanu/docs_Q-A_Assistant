import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    def __init__(self):
        # Database
        self.DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'solstice_docs.db'}")
        
        # Google Gemini API
        self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
        self.GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.GEMINI_GENERATION_MODEL: str = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
        
        # Retrieval & Chunking
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
        self.RRF_K: int = int(os.getenv("RRF_K", "60"))
        self.TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "3"))
        
        # Evidence Gate Thresholds
        self.BM25_SCORE_FLOOR: float = float(os.getenv("BM25_SCORE_FLOOR", "0.05"))
        self.COSINE_SCORE_FLOOR: float = float(os.getenv("COSINE_SCORE_FLOOR", "0.40"))
        
        # Fallback text
        self.FALLBACK_ANSWER: str = "I don't have an answer for that."
        
        # Starter data path
        self.DEFAULT_DOCS_JSON_PATH: Path = BASE_DIR / "starter_kit" / "data" / "docs.json"

settings = Settings()
