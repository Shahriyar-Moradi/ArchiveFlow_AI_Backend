"""
Configuration management for FastAPI Document Automation Backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

class Settings:
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Document Automation System"
    VERSION: str = "1.0.0"
    
    # Anthropic API Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    
    # Google Cloud Storage Configuration
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "voucher-bucket-1")
    GCS_PROJECT_ID: str = os.getenv("GCS_PROJECT_ID", "rocasoft")
    
    # Optional: Service account key path (defaults to file in current dir, falls back to ADC)
    GCS_SERVICE_ACCOUNT_KEY: str = os.getenv(
        "GCS_SERVICE_ACCOUNT_KEY",
        str(Path(__file__).parent / "voucher-storage-key.json")
    )
    
    # Firestore Configuration
    FIRESTORE_PROJECT_ID: str = os.getenv("FIRESTORE_PROJECT_ID", "rocasoft")
    FIRESTORE_COLLECTION_DOCUMENTS: str = "documents"
    FIRESTORE_COLLECTION_JOBS: str = "processing_jobs"
    FIRESTORE_COLLECTION_FLOWS: str = "flows"
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".png", ".jpg", ".jpeg"}
    TEMP_UPLOAD_FOLDER: str = "temp"
    ORGANIZED_FOLDER: str = "organized_vouchers"
    
    # Processing Configuration
    OCR_MAX_RETRIES: int = 3
    OCR_RETRY_DELAY: int = 15  # seconds (reduced from 30 for faster retries)
    
    # CORS Configuration
    # Allow origins for web, mobile, and Capacitor apps
    CORS_ORIGINS: list = [
        origin.strip() 
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8080,http://localhost:4200,capacitor://localhost,ionic://localhost,http://localhost,https://localhost,https://docflowai-c88e6.web.app,https://docflowai-c88e6.firebaseapp.com"
        ).split(",")
        if origin.strip()  # Remove empty strings
    ]
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))
    
    # Mock Configuration
    USE_MOCK_SERVICES: bool = os.getenv("USE_MOCK_SERVICES", "false").lower() == "true"
    
    # Performance Optimization
    SKIP_CLASSIFICATION: bool = os.getenv("SKIP_CLASSIFICATION", "false").lower() == "true"  # Skip classification step for speed
    
    @property
    def anthropic_api_key_configured(self) -> bool:
        """Check if Anthropic API key is configured"""
        return bool(self.ANTHROPIC_API_KEY)
    
    @property
    def gcs_configured(self) -> bool:
        """Check if GCS is configured"""
        return bool(self.GCS_BUCKET_NAME and self.GCS_PROJECT_ID)

settings = Settings()

