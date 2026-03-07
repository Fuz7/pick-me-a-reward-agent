import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    
    # LLM
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")

    # GROQ API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Database
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

settings = Settings()