"""
Centralized Configuration Settings for AI Data Analyst Agent.
Handles environment variables, storage paths, security settings, DoS safeguards, and defaults.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger(__name__)


class Config:
    """Application configuration class with strict security and production defaults."""

    # Base Directory
    BASE_DIR = BASE_DIR

    # Flask Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development").lower()
    
    # In production, debug mode MUST be disabled by default
    _raw_debug = os.getenv("FLASK_DEBUG", None)
    if _raw_debug is not None:
        DEBUG = _raw_debug.lower() in ("true", "1", "yes")
    else:
        DEBUG = FLASK_ENV != "production"

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # File Upload & Storage Settings
    MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", 50))
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH_MB * 1024 * 1024  # Enforced in Flask
    ALLOWED_EXTENSIONS = set(
        ext.strip().lower() for ext in os.getenv("ALLOWED_EXTENSIONS", "csv").split(",")
    )

    # Storage Paths (relative to project root or absolute paths)
    UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "data/uploads")
    PROCESSED_FOLDER = BASE_DIR / os.getenv("PROCESSED_FOLDER", "data/processed")
    REPORTS_FOLDER = BASE_DIR / os.getenv("REPORTS_FOLDER", "data/reports")
    TEMP_FOLDER = BASE_DIR / os.getenv("TEMP_FOLDER", "data/temp")

    # Retention & Pruning Safeguards
    FILE_RETENTION_HOURS = int(os.getenv("FILE_RETENTION_HOURS", 24))

    # DoS & Resource Exhaustion Protection Limits
    MAX_ROWS_LIMIT = int(os.getenv("MAX_ROWS_LIMIT", 1_000_000))
    MAX_COLUMNS_LIMIT = int(os.getenv("MAX_COLUMNS_LIMIT", 500))
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 5000))

    # Supported Encodings for Resilient Parsing
    FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

    # AI Configuration
    AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")

    @classmethod
    def is_production(cls) -> bool:
        """Check if application is running in production mode."""
        return cls.FLASK_ENV == "production"

    @classmethod
    def validate_production_config(cls) -> List[str]:
        """
        Validate security settings for production deployments.
        Returns a list of warnings or recommendations.
        """
        warnings = []
        if cls.is_production():
            if cls.DEBUG:
                warnings.append("SECURITY WARNING: DEBUG mode is active in production environment.")
            if cls.SECRET_KEY in ("dev-secret-key-change-in-production", "replace-with-a-secure-random-secret-key-in-production", ""):
                warnings.append("SECURITY CRITICAL: Default insecure SECRET_KEY detected in production. Set a unique, cryptographically strong SECRET_KEY.")
            if cls.HOST in ("0.0.0.0", "::") and not os.getenv("REVERSE_PROXY_CONFIGURED"):
                warnings.append("SECURITY NOTICE: Application is bound to all interfaces. Ensure a reverse proxy (Nginx, Traefik) is configured with TLS.")
        return warnings

    @classmethod
    def get_security_summary(cls) -> dict:
        """
        Return high-level summary of active security controls,
        DoS limits, and environment modes without exposing credentials.
        """
        return {
            "environment": cls.FLASK_ENV,
            "debug_mode": cls.DEBUG,
            "is_production": cls.is_production(),
            "max_upload_size_mb": cls.MAX_CONTENT_LENGTH_MB,
            "allowed_extensions": sorted(list(cls.ALLOWED_EXTENSIONS)),
            "max_rows_limit": cls.MAX_ROWS_LIMIT,
            "max_columns_limit": cls.MAX_COLUMNS_LIMIT,
            "file_retention_hours": cls.FILE_RETENTION_HOURS,
            "configured_ai_providers": [k for k, v in cls.get_available_providers().items() if v.get("configured")],
        }

    @classmethod
    def get_available_providers(cls) -> dict:
        """
        Check which AI providers are configured with valid API keys.
        Never exposes the actual secret keys.
        """
        providers = {
            "gemini": {
                "name": "Google Gemini",
                "configured": bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY.strip() and not cls.GEMINI_API_KEY.startswith("your_")),
                "default_model": os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
            },
            "openai": {
                "name": "OpenAI",
                "configured": bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY.strip() and not cls.OPENAI_API_KEY.startswith("your_")),
                "default_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "configured": bool(cls.ANTHROPIC_API_KEY and cls.ANTHROPIC_API_KEY.strip() and not cls.ANTHROPIC_API_KEY.startswith("your_")),
                "default_model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
            },
            "deterministic_fallback": {
                "name": "Deterministic Engine (No API Key Required)",
                "configured": True,
                "default_model": "rule-based-synthesizer",
                "models": ["rule-based-synthesizer"],
            },
        }
        return providers

    @classmethod
    def init_app(cls):
        """Ensure all required storage directories exist."""
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.TEMP_FOLDER.mkdir(parents=True, exist_ok=True)


# Initialize folders upon import
Config.init_app()
