import logging
from logging.handlers import RotatingFileHandler
import os
from .config import settings

# ==========================================
# Custom Exception Classes
# ==========================================

class AppError(Exception):
    """Base exception for application errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class DocumentProcessingError(AppError):
    """Raised when document ingestion/processing fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=422)

class RAGError(AppError):
    """Raised when RAG/Vector store operations fail"""
    def __init__(self, message: str):
        super().__init__(message, status_code=503)

class DatabaseError(AppError):
    """Raised when database operations fail"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

class ThreadError(AppError):
    """Raised when thread operations fail"""
    def __init__(self, message: str, status_code: int = 404):
        super().__init__(message, status_code=status_code)

# ==========================================
# Logging Configuration
# ==========================================

def setup_logger(name: str):
    """Configure logger with console and file output"""
    
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)

    
    # Prevent duplicate logs if logger already exists
    if logger.hasHandlers():
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        "logs/app.log", maxBytes=10485760, backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
