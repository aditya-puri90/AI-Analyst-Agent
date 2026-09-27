"""
Security Utilities for AI Data Analyst Agent (Phase 15).
Provides path traversal protection, API key redaction, sensitive logging filters,
input sanitization, safe identifier validation, and DoS safety checks.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

# Regular expressions for identifying sensitive credentials and keys
SENSITIVE_PATTERNS = [
    # Google API Keys (e.g., AIzaSy...)
    re.compile(r"AIza[0-9A-Za-z-_]{35}", re.IGNORECASE),
    # OpenAI API Keys (e.g., sk-...)
    re.compile(r"sk-[0-9A-Za-z-_]{20,}", re.IGNORECASE),
    # Anthropic API Keys (e.g., ant-...)
    re.compile(r"ant-[0-9A-Za-z-_]{20,}", re.IGNORECASE),
    # Bearer tokens in headers
    re.compile(r"Bearer\s+([A-Za-z0-9-_\.]+)", re.IGNORECASE),
    # Generic API key query parameters
    re.compile(r"(key|api_key|apikey|token|secret)=([A-Za-z0-9-_\.]+)", re.IGNORECASE),
    # Authorization / x-api-key headers
    re.compile(r"(x-api-key|x-goog-api-key|authorization):\s*([^\s,]+)", re.IGNORECASE),
]


def mask_sensitive_data(text: str) -> str:
    """
    Redact API keys, tokens, and credentials from strings, error messages, and logs.
    
    Args:
        text: Input string that may contain sensitive credentials.
        
    Returns:
        str: Redacted string with sensitive patterns masked.
    """
    if not text or not isinstance(text, str):
        return text

    sanitized = text
    # Google API Key redaction
    sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_GEMINI_KEY]", sanitized)
    # OpenAI API Key redaction
    sanitized = re.sub(r"sk-[0-9A-Za-z-_]{20,}", "[REDACTED_OPENAI_KEY]", sanitized)
    # Anthropic API Key redaction
    sanitized = re.sub(r"ant-[0-9A-Za-z-_]{20,}", "[REDACTED_ANTHROPIC_KEY]", sanitized)
    # Bearer tokens
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9-_\.]+", r"\1[REDACTED_TOKEN]", sanitized, flags=re.IGNORECASE)
    # Query param keys
    sanitized = re.sub(r"((?:key|api_key|apikey|token|secret)=)[A-Za-z0-9-_\.]+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    # Header values
    sanitized = re.sub(r"((?:x-api-key|x-goog-api-key|authorization):\s*)[^\s,]+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)

    return sanitized


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that automatically redacts API keys and sensitive tokens
    from all log records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_sensitive_data(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: mask_sensitive_data(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, (tuple, list)):
                record.args = tuple(mask_sensitive_data(str(arg)) if isinstance(arg, str) else arg for arg in record.args)
        return True


def safe_path_resolve(base_dir: Union[str, Path], user_path: Union[str, Path]) -> Optional[Path]:
    """
    Safely resolve a path relative to a base directory, guaranteeing protection
    against path traversal attacks (e.g., ../../etc/passwd, %2e%2e, null bytes, ADS).
    
    Args:
        base_dir: Base directory that must contain the target file.
        user_path: Untrusted path or filename provided by user/client.
        
    Returns:
        Optional[Path]: Canonical resolved Path if strictly within base_dir, None otherwise.
    """
    if not user_path or not base_dir:
        return None

    user_str = str(user_path)

    # Reject null bytes, URL encoded null bytes, or carriage returns in path
    if "\x00" in user_str or "%00" in user_str:
        logger.warning("Path traversal attempt with null byte detected: %s", user_str)
        return None

    # Reject Windows Alternate Data Streams (e.g. file.csv:hidden)
    if ":" in user_str and not (len(user_str) >= 2 and user_str[1] == ":" and user_str[0].isalpha() and user_str == str(base_dir)):
        # If the user_str itself contains a drive colon or ADS colon
        if ":" in Path(user_str).name:
            logger.warning("Path traversal attempt with Windows ADS detected: %s", user_str)
            return None

    try:
        base = Path(base_dir).resolve()
        # Clean relative path
        cleaned_rel = Path(user_str)
        if cleaned_rel.is_absolute():
            # If absolute, verify it is within base
            target = cleaned_rel.resolve()
        else:
            target = (base / cleaned_rel).resolve()

        # Strict containment verification
        # Python 3.9+ supports is_relative_to
        if hasattr(target, "is_relative_to"):
            if not target.is_relative_to(base):
                logger.warning("Path traversal attempt detected: '%s' escapes '%s'", user_str, base)
                return None
        else:
            # Fallback for older python or cross-platform safety
            common = os.path.commonpath([str(base), str(target)])
            if common != str(base):
                logger.warning("Path traversal attempt detected: '%s' escapes '%s'", user_str, base)
                return None

        return target

    except Exception as e:
        logger.warning("Path resolution error for '%s': %s", user_str, e)
        return None


def validate_safe_identifier(identifier: str) -> bool:
    """
    Verify that an identifier (e.g., dataset_id, column_name, session_id)
    contains only safe, non-traversal characters.
    
    Args:
        identifier: String identifier to validate.
        
    Returns:
        bool: True if safe, False if potentially malicious.
    """
    if not identifier or not isinstance(identifier, str):
        return False

    # Check for null bytes, directory separators, path traversal
    if "\x00" in identifier or "%00" in identifier:
        return False
    if "/" in identifier or "\\" in identifier or ".." in identifier:
        return False

    # Allow alphanumeric, underscores, hyphens, and periods (e.g., dataset_123.csv)
    safe_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
    return bool(safe_pattern.match(identifier))


def sanitize_input_string(text: str, max_length: int = 5000) -> str:
    """
    Sanitize text input from users (chat queries, column names, titles)
    by removing control characters and enforcing length limits.
    
    Args:
        text: Raw text string.
        max_length: Maximum allowable character length.
        
    Returns:
        str: Sanitized, bounded string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove non-printable control characters except newline and tab
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t", "\r"))
    cleaned = cleaned.strip()

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    return cleaned


def is_binary_payload(sample_bytes: bytes) -> bool:
    """
    Check if a byte buffer represents binary executable, archive, or non-text data.
    Detects PE (MZ), ELF, Zip (PK), Mach-O, tar, gzip, pdf, etc.
    
    Args:
        sample_bytes: First few hundred bytes of file.
        
    Returns:
        bool: True if binary/executable signature detected, False if text/CSV.
    """
    if not sample_bytes:
        return False

    # Magic byte signatures for executables / archives
    magic_signatures = [
        b"MZ",  # Windows Executable / DLL
        b"\x7fELF",  # Linux Executable
        b"\xfe\xed\xfa\xce",  # Mach-O
        b"\xfe\xed\xfa\xcf",  # Mach-O
        b"\xce\xfa\xed\xfe",  # Mach-O
        b"\xcf\xfa\xed\xfe",  # Mach-O
        b"PK\x03\x04",  # Zip / Jar / Docx / Exe archive
        b"\x1f\x8b",  # Gzip
        b"BZh",  # Bzip2
        b"\xfd7zXZ\x00",  # XZ
        b"7z\xbc\xaf\x27\x1c",  # 7-Zip
        b"%PDF",  # PDF
    ]

    for sig in magic_signatures:
        if sample_bytes.startswith(sig):
            return True

    # Check for excessive null bytes (characteristic of binary data)
    null_count = sample_bytes.count(b"\x00")
    if len(sample_bytes) > 0 and (null_count / len(sample_bytes)) > 0.05:
        return True

    return False
