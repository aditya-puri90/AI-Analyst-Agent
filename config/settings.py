"""
Centralized Configuration Settings for AI Data Analyst Agent.
Handles environment variables, storage paths, security settings, and defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Application configuration class."""

    # Flask Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))

    # File Upload & Storage Settings
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", 50)) * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS = set(
        ext.strip().lower() for ext in os.getenv("ALLOWED_EXTENSIONS", "csv").split(",")
    )

    UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "data/uploads")
    PROCESSED_FOLDER = BASE_DIR / os.getenv("PROCESSED_FOLDER", "data/processed")

    # Supported Encodings for Resilient Parsing
    FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

    # AI Configuration (Phase 9: AI Insight Engine)
    AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")

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
        """Ensure necessary storage directories exist."""
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


# Initialize folders upon import
Config.init_app()

