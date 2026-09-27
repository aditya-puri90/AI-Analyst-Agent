"""
Validation utilities for file uploads and CSV structural integrity.
Enforces file security, size caps, encoding resilience, delimiter sniffing, and data validation.
"""

import csv
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from werkzeug.utils import secure_filename

from config.settings import Config
from utils.security import is_binary_payload

logger = logging.getLogger(__name__)

# Extensions considered dangerous when chained (e.g. upload.php.csv, script.sh.csv)
DANGEROUS_INTERMEDIATE_EXTENSIONS = {
    "exe", "dll", "so", "sh", "bat", "cmd", "ps1", "vbs", "js", "ts",
    "py", "phtml", "php", "php3", "php4", "php5", "phps", "cgi", "pl",
    "jar", "war", "ear", "asp", "aspx", "jsp", "jspx", "htm", "html",
    "shtml", "svg", "xml", "bin", "elf", "scr"
}


def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension (.csv) and is free
    from malicious double-extensions or hidden suffixes.
    
    Args:
        filename: Name of the uploaded file.
        
    Returns:
        bool: True if allowed and safe, False otherwise.
    """
    if not filename or not isinstance(filename, str):
        return False

    # Reject null bytes or URL encoded null bytes
    if "\x00" in filename or "%00" in filename:
        return False

    # Strip whitespace
    clean_name = filename.strip()
    if "." not in clean_name:
        return False

    # Check for hidden files starting with dot
    if clean_name.startswith("."):
        return False

    # Split parts
    parts = clean_name.lower().split(".")
    ext = parts[-1].strip()

    # Final extension must be in allowed list
    if ext not in Config.ALLOWED_EXTENSIONS:
        return False

    # Check for dangerous multi-part extensions (e.g., evil.php.csv, script.sh.csv)
    if len(parts) > 2:
        for intermediate in parts[1:-1]:
            if intermediate in DANGEROUS_INTERMEDIATE_EXTENSIONS:
                logger.warning("Rejected file with dangerous intermediate extension: '%s'", filename)
                return False

    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal or unsafe characters.
    
    Args:
        filename: Raw filename string.
        
    Returns:
        str: Safe, sanitized filename.
    """
    if not filename or not isinstance(filename, str):
        return "dataset.csv"

    # Remove null bytes
    cleaned = filename.replace("\x00", "").replace("%00", "")
    # Use werkzeug secure_filename
    cleaned = secure_filename(cleaned)

    if not cleaned or cleaned.startswith(".") or cleaned == "csv":
        cleaned = "dataset.csv"

    if not cleaned.lower().endswith(".csv"):
        cleaned += ".csv"

    # Limit filename length
    if len(cleaned) > 100:
        cleaned = cleaned[:96] + ".csv"

    return cleaned


def detect_encoding_and_delimiter(file_path: Path) -> Tuple[str, str]:
    """
    Resiliently detect the text encoding and CSV delimiter of a file.
    Tests candidate encodings: utf-8, utf-8-sig, latin-1, cp1252, iso-8859-1.
    
    Args:
        file_path: Path to the target CSV file.
        
    Returns:
        Tuple[str, str]: (detected_encoding, detected_delimiter)
    """
    detected_encoding = "utf-8"
    sample_bytes = b""

    # Try candidate encodings in priority order
    for encoding in Config.FALLBACK_ENCODINGS:
        try:
            with open(file_path, "rb") as f:
                sample_bytes = f.read(65536)  # 64KB sample
            sample_bytes.decode(encoding)
            detected_encoding = encoding
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Delimiter detection
    detected_delimiter = ","
    try:
        sample_text = sample_bytes.decode(detected_encoding)
        if sample_text.strip():
            # Use Python's csv.Sniffer
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text[:8192], delimiters=[",", ";", "\t", "|"])
            detected_delimiter = dialect.delimiter
    except Exception as e:
        logger.debug("CSV Sniffer delimiter detection fallback: %s", e)
        # Fallback: Count delimiters in the first non-empty lines
        try:
            lines = [line for line in sample_text.splitlines() if line.strip()]
            if lines:
                candidate_counts = {d: lines[0].count(d) for d in [",", ";", "\t", "|"]}
                best_delimiter = max(candidate_counts, key=candidate_counts.get)
                if candidate_counts[best_delimiter] > 0:
                    detected_delimiter = best_delimiter
        except Exception:
            detected_delimiter = ","

    return detected_encoding, detected_delimiter


def validate_file_upload(file_storage) -> Tuple[bool, Optional[str]]:
    """
    Validate an incoming Flask FileStorage object before saving to disk.
    Enforces format, extension, size, empty checks, and binary payload detection.
    
    Args:
        file_storage: Werkzeug FileStorage instance.
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if file_storage is None or not getattr(file_storage, "filename", None):
        return False, "No file was selected for upload. Please choose a CSV file."

    if not allowed_file(file_storage.filename):
        allowed_str = ", ".join([f".{ext}" for ext in Config.ALLOWED_EXTENSIONS])
        return False, f"Invalid file format '{file_storage.filename}'. Only CSV files ({allowed_str}) are supported."

    # Check for empty content (0 bytes) and inspect size
    file_storage.seek(0, 2)  # Seek to end of file
    file_size = file_storage.tell()
    file_storage.seek(0)  # Reset stream position to beginning

    if file_size == 0:
        return False, "The uploaded file is empty (0 bytes). Please upload a valid CSV containing data."

    if file_size > Config.MAX_CONTENT_LENGTH:
        max_mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        return False, f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."

    # Inspect first 512 bytes for binary/executable payload disguised as CSV
    sample_header = b""
    if hasattr(file_storage, "read") and callable(getattr(file_storage, "read")):
        try:
            sample_header = file_storage.read(512)
            if hasattr(file_storage, "seek"):
                file_storage.seek(0)  # Reset stream position
        except Exception:
            sample_header = b""

    if sample_header and is_binary_payload(sample_header):
        logger.warning("Rejected binary/executable payload upload for filename '%s'", file_storage.filename)
        return False, "Binary or executable files are not supported. Please upload a valid plain-text CSV file."

    return True, None


def validate_csv_structure(file_path: Path) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Verify that the CSV file can be properly decoded and contains valid tabular data.
    Checks for:
    - Empty or whitespace-only files
    - Binary payload signatures
    - Missing or all-blank header rows
    - Absence of data rows (header-only CSV)
    - Structural corruption (mismatched quotes, unparseable lines)
    - DoS limits on maximum allowable columns and rows
    
    Args:
        file_path: Path to the saved CSV file.
        
    Returns:
        Tuple[bool, Optional[str], Optional[Dict[str, Any]]]: (is_valid, error_message, metadata)
    """
    if not file_path.exists():
        return False, f"File does not exist at path: {file_path}", None

    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "The uploaded CSV file is empty (0 bytes).", None

    try:
        # Check binary signature
        with open(file_path, "rb") as bf:
            sample_start = bf.read(512)
        if is_binary_payload(sample_start):
            return False, "File contains binary or executable content. Only plain-text CSV files are permitted.", None

        encoding, delimiter = detect_encoding_and_delimiter(file_path)

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            # Read first row as header
            try:
                header = next(reader)
            except StopIteration:
                return False, "CSV file is completely empty.", None
            except csv.Error as e:
                return False, f"CSV parsing error in header row: {str(e)}", None

            if not header or all(not str(col).strip() for col in header):
                return False, "CSV header row is missing or contains only empty column names.", None

            # Clean header column names
            clean_headers = [str(col).strip() for col in header]
            expected_cols = len(clean_headers)

            if expected_cols > Config.MAX_COLUMNS_LIMIT:
                return False, f"Dataset contains {expected_cols} columns, exceeding the maximum allowed limit of {Config.MAX_COLUMNS_LIMIT}.", None

            # Count rows and check for corruption
            row_count = 0
            inconsistent_rows = 0

            for line_idx, row in enumerate(reader, start=2):
                row_count += 1
                if len(row) != expected_cols:
                    inconsistent_rows += 1
                
                # Check DoS row count limit
                if row_count > Config.MAX_ROWS_LIMIT:
                    return False, f"Dataset exceeds the maximum allowed row limit of {Config.MAX_ROWS_LIMIT:,} rows.", None

                # Sample up to first 500 rows for deep integrity validation
                if line_idx > 500:
                    for _ in reader:
                        row_count += 1
                        if row_count > Config.MAX_ROWS_LIMIT:
                            return False, f"Dataset exceeds the maximum allowed row limit of {Config.MAX_ROWS_LIMIT:,} rows.", None
                    break

            if row_count == 0:
                return False, "The CSV file contains a header row but has no data rows. At least 1 row of data is required.", None

        metadata = {
            "encoding": encoding,
            "delimiter": delimiter,
            "column_count": expected_cols,
            "row_count": row_count,
            "approx_row_count": row_count,
            "columns": clean_headers,
            "inconsistent_rows_count": inconsistent_rows,
        }

        return True, None, metadata

    except UnicodeError as ue:
        logger.error("Encoding error validating CSV: %s", ue)
        return False, "Encoding error: Unable to decode file. Supported encodings are UTF-8, Latin-1, CP1252.", None
    except csv.Error as ce:
        logger.error("CSV structural syntax error: %s", ce)
        return False, f"CSV formatting error: {str(ce)}. Please check for unescaped quotes or corrupted lines.", None
    except Exception as e:
        logger.error("CSV structural validation failed: %s", e, exc_info=True)
        return False, f"Failed to parse CSV: {str(e)}", None
