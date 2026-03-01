# backend/config.py

from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Field
import os


class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = Field(..., alias="API_KEY")
    HUGGING_FACE_API_KEY: Optional[str] = Field(None, alias="Hugging_Face_API")
    ALLOWED_ORIGINS: list = ["*"]
    
    #auth
    JWT_SECRET_KEY : str = Field(...,alias='JWT_SECRET_KEY')
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    
    # Database
    SQLITE_DB_PATH: str = "./chat_database.db"
    CHROMA_DB_PATH: str = "./chroma_db"
    UPLOAD_DIR: str = "./uploads"
    
    # Models
    GEMINI_MODEL_NAME: str = "models/gemini-2.5-flash"
    GROK_MODEL_NAME: str = "llama-3.1-8b-instant"
    #EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # Limits
    MAX_FILE_SIZE_MB: int = 10
    MAX_CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 150
    MAX_REQUESTS_PER_MINUTE: str = "7/day"
    RATE_LIMIT_SECONDS : int = 86400
    
    # Retrieval
    RETRIEVAL_K: int = 4
    RETRIEVAL_FETCH_K: int = 12
    RETRIEVAL_LAMBDA: float = 0.7
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore" 

settings = Settings()